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
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        self.block_size = block_size
        self.micro_batch_size = micro_batch_size
        self.device = device

        self.doc_tokenize_batch = doc_tokenize_batch
        self.max_doc_tokens = max_doc_tokens
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        self.prefetch_batches = prefetch_batches
        self.queue = queue.Queue(maxsize=prefetch_batches)

        self.stop_event = threading.Event()
        self.worker = None

        self.last_wait_time = 0.0

    def start(self):
        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self.worker.start()

        # Чуть-чуть ждём, чтобы очередь начала заполняться
        while self.queue.qsize() == 0:
            time.sleep(0.05)

    def stop(self):
        self.stop_event.set()

        if self.worker is not None:
            self.worker.join(timeout=2.0)

    def _make_dataset_iter(self):
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

        return iter(dataset)

    def _worker_loop(self):
        dataset_iter = self._make_dataset_iter()

        token_buffer = []
        cursor = 0

        tokens_per_batch = self.micro_batch_size * (self.block_size + 1)

        def available_tokens():
            return len(token_buffer) - cursor

        def compact_if_needed():
            nonlocal token_buffer, cursor
            if cursor > 1_000_000 or cursor > len(token_buffer) // 2:
                token_buffer = token_buffer[cursor:]
                cursor = 0

        def refill(min_tokens):
            nonlocal dataset_iter, token_buffer

            while available_tokens() < min_tokens and not self.stop_event.is_set():
                texts = []

                while len(texts) < self.doc_tokenize_batch:
                    try:
                        sample = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = self._make_dataset_iter()
                        sample = next(dataset_iter)

                    text = sample.get("text", None)

                    if isinstance(text, str) and len(text) > 50:
                        texts.append(text)

                encoded = self.tokenizer(
                    texts,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_doc_tokens,
                )

                for ids in encoded["input_ids"]:
                    if len(ids) > 0:
                        token_buffer.extend(ids)
                        token_buffer.append(self.eos_id)

        while not self.stop_event.is_set():
            try:
                refill(tokens_per_batch * 8)

                chunk = token_buffer[cursor:cursor + tokens_per_batch]
                cursor += tokens_per_batch

                compact_if_needed()

                if len(chunk) != tokens_per_batch:
                    continue

                data_cpu = torch.tensor(
                    chunk,
                    dtype=torch.long,
                ).view(self.micro_batch_size, self.block_size + 1)

                if self.device.type == "cuda":
                    data_cpu = data_cpu.pin_memory()

                self.queue.put(data_cpu, timeout=1.0)

            except queue.Full:
                continue
            except Exception as e:
                print("Async batcher worker error:", repr(e))
                time.sleep(1.0)

    def get_batch(self):
        t0 = time.time()
        data_cpu = self.queue.get()
        self.last_wait_time = time.time() - t0

        data = data_cpu.to(self.device, non_blocking=True)

        x = data[:, :-1].contiguous()
        y = data[:, 1:].contiguous()

        return x, y

    def qsize(self):
        return self.queue.qsize()