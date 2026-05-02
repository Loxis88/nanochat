import time
import queue
import threading
import torch
from datasets import load_dataset


class AsyncFineWebBatcher:
    def __init__(
        self,
        dataset_name,
        dataset_config,
        tokenizer,
        eos_id,
        block_size,
        micro_batch_size,
        device,
        doc_tokenize_batch=512,
        max_doc_tokens=4096,
        shuffle_buffer=10_000,
        seed=42,
        prefetch_batches=32,
        skip_tokens=0,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        self.block_size = block_size
        self.micro_batch_size = micro_batch_size
        self.device = torch.device(device)

        self.doc_tokenize_batch = doc_tokenize_batch
        self.max_doc_tokens = max_doc_tokens
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        self.prefetch_batches = prefetch_batches
        self.queue = queue.Queue(maxsize=prefetch_batches)

        self.stop_event = threading.Event()
        self.worker = None

        self.last_wait_time = 0.0

        # Dataset state
        self.dataset = None
        self.dataset_iter = None
        self.dataset_state_to_load = None

        # Token stream state
        self.token_buffer = []
        self.cursor = 0

        # Approximate resume fallback
        self.skip_tokens = int(skip_tokens)
        self.skipped_tokens = 0

        # Restored ready batches
        self.restored_queue = []

        # Batch currently produced but maybe not queued yet
        self.inflight_batch = None

        # Stats
        self.total_docs_seen = 0
        self.total_tokens_produced = 0
        self.total_batches_produced = 0
        self.total_batches_consumed = 0

        self.state_lock = threading.RLock()

    def _make_dataset(self):
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
        )

        dataset = dataset.shuffle(
            buffer_size=self.shuffle_buffer,
            seed=self.seed,
        )

        if self.dataset_state_to_load is not None:
            dataset.load_state_dict(self.dataset_state_to_load)
            self.dataset_state_to_load = None

            # Если есть точный dataset_state, skip_tokens больше не нужен.
            self.skip_tokens = 0
            self.skipped_tokens = 0

        return dataset

    def _ensure_dataset_iter_locked(self):
        if self.dataset is None:
            self.dataset = self._make_dataset()
            self.dataset_iter = iter(self.dataset)

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return

        self.stop_event.clear()

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self.worker.start()

        while self.queue.qsize() == 0 and not self.stop_event.is_set():
            time.sleep(0.05)

    def stop(self):
        self.stop_event.set()

        if self.worker is not None:
            self.worker.join(timeout=5.0)

    def _available_tokens_locked(self):
        return len(self.token_buffer) - self.cursor

    def _compact_if_needed_locked(self):
        if self.cursor > 1_000_000 or self.cursor > len(self.token_buffer) // 2:
            self.token_buffer = self.token_buffer[self.cursor:]
            self.cursor = 0

    def _append_or_skip_tokens_locked(self, ids):
        if len(ids) == 0:
            return

        ids = list(ids)
        ids.append(self.eos_id)

        if self.skip_tokens > 0 and self.skipped_tokens < self.skip_tokens:
            need = self.skip_tokens - self.skipped_tokens

            if need >= len(ids):
                self.skipped_tokens += len(ids)
                return

            ids = ids[need:]
            self.skipped_tokens = self.skip_tokens

        self.token_buffer.extend(ids)

    def _refill_locked(self, min_tokens):
        self._ensure_dataset_iter_locked()

        while self._available_tokens_locked() < min_tokens and not self.stop_event.is_set():
            texts = []

            while len(texts) < self.doc_tokenize_batch and not self.stop_event.is_set():
                try:
                    sample = next(self.dataset_iter)
                except StopIteration:
                    self.dataset = self._make_dataset()
                    self.dataset_iter = iter(self.dataset)
                    sample = next(self.dataset_iter)

                self.total_docs_seen += 1

                text = sample.get("text", None)

                if isinstance(text, str) and len(text) > 50:
                    texts.append(text)

            if not texts:
                continue

            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_doc_tokens,
            )

            for ids in encoded["input_ids"]:
                self._append_or_skip_tokens_locked(ids)

    def _make_one_batch_locked(self):
        tokens_per_batch = self.micro_batch_size * (self.block_size + 1)

        self._refill_locked(tokens_per_batch * 8)

        chunk = self.token_buffer[self.cursor:self.cursor + tokens_per_batch]

        if len(chunk) != tokens_per_batch:
            return None

        self.cursor += tokens_per_batch
        self._compact_if_needed_locked()

        data_cpu = torch.tensor(
            chunk,
            dtype=torch.long,
        ).view(self.micro_batch_size, self.block_size + 1)

        if self.device.type == "cuda":
            data_cpu = data_cpu.pin_memory()

        self.total_tokens_produced += tokens_per_batch
        self.total_batches_produced += 1

        return data_cpu

    def _worker_loop(self):
        while self.restored_queue and not self.stop_event.is_set():
            data_cpu = self.restored_queue.pop(0)

            if self.device.type == "cuda" and not data_cpu.is_pinned():
                data_cpu = data_cpu.pin_memory()

            try:
                self.queue.put(data_cpu, timeout=1.0)
            except queue.Full:
                self.restored_queue.insert(0, data_cpu)
                time.sleep(0.05)

        while not self.stop_event.is_set():
            try:
                with self.state_lock:
                    data_cpu = self._make_one_batch_locked()
                    self.inflight_batch = data_cpu

                if data_cpu is None:
                    continue

                self.queue.put(data_cpu, timeout=1.0)

                with self.state_lock:
                    if self.inflight_batch is data_cpu:
                        self.inflight_batch = None

            except queue.Full:
                continue

            except Exception as e:
                print("Async batcher worker error:", repr(e))
                time.sleep(1.0)

    def get_batch(self):
        t0 = time.time()
        data_cpu = self.queue.get()
        self.last_wait_time = time.time() - t0

        self.total_batches_consumed += 1

        data = data_cpu.to(self.device, non_blocking=True)

        x = data[:, :-1].contiguous()
        y = data[:, 1:].contiguous()

        return x, y

    def qsize(self):
        return self.queue.qsize()

    def state_dict(self):
        with self.state_lock:
            dataset_state = None

            if self.dataset is not None and hasattr(self.dataset, "state_dict"):
                try:
                    dataset_state = self.dataset.state_dict()
                except Exception as e:
                    print("Warning: dataset.state_dict() failed:", repr(e))
                    dataset_state = None

            token_tail = list(self.token_buffer[self.cursor:])

            inflight = self.inflight_batch

            meta = {
                "total_docs_seen": self.total_docs_seen,
                "total_tokens_produced": self.total_tokens_produced,
                "total_batches_produced": self.total_batches_produced,
                "total_batches_consumed": self.total_batches_consumed,
                "skip_tokens": self.skip_tokens,
                "skipped_tokens": self.skipped_tokens,
            }

        with self.queue.mutex:
            queue_items = list(self.queue.queue)
            queue_ids = {id(x) for x in queue_items}

            queued_batches = [
                item.detach().cpu().clone()
                for item in queue_items
            ]

        if inflight is not None and id(inflight) not in queue_ids:
            queued_batches.append(inflight.detach().cpu().clone())

        return {
            "dataset_state": dataset_state,
            "token_buffer": token_tail,
            "cursor": 0,
            "queued_batches": queued_batches,

            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "eos_id": self.eos_id,
            "block_size": self.block_size,
            "micro_batch_size": self.micro_batch_size,
            "doc_tokenize_batch": self.doc_tokenize_batch,
            "max_doc_tokens": self.max_doc_tokens,
            "shuffle_buffer": self.shuffle_buffer,
            "seed": self.seed,
            "prefetch_batches": self.prefetch_batches,

            "meta": meta,
        }

    def load_state_dict(self, state):
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("load_state_dict() надо вызывать ДО start().")

        self.dataset_state_to_load = state.get("dataset_state", None)

        self.token_buffer = list(state.get("token_buffer", []))
        self.cursor = int(state.get("cursor", 0))

        self.restored_queue = [
            item.detach().cpu().clone()
            for item in state.get("queued_batches", [])
        ]

        meta = state.get("meta", {})

        self.total_docs_seen = int(meta.get("total_docs_seen", 0))
        self.total_tokens_produced = int(meta.get("total_tokens_produced", 0))
        self.total_batches_produced = int(meta.get("total_batches_produced", 0))
        self.total_batches_consumed = int(meta.get("total_batches_consumed", 0))

        if self.dataset_state_to_load is not None:
            self.skip_tokens = 0
            self.skipped_tokens = 0

        self.dataset = None
        self.dataset_iter = None
        self.inflight_batch = None