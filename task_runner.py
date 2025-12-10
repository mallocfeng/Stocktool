from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Optional


TaskWorker = Callable[[threading.Event, Callable[[str, object], None]], None]
TaskCallback = Callable[[object], None]


@dataclass
class _Task:
    name: str
    thread: threading.Thread
    queue: Queue
    stop_event: threading.Event
    on_result: TaskCallback
    on_error: TaskCallback
    on_finished: Callable[[], None]


class TaskRunner:
    def __init__(
        self,
        root,
        status_callback: Callable[[str], None],
        log_callback: Callable[[str], None],
        poll_interval: int = 100,
    ):
        self.root = root
        self.status_callback = status_callback
        self.log_callback = log_callback
        self.poll_interval = poll_interval
        self.current: Optional[_Task] = None
        self.pending: Optional[Callable[[], None]] = None

    def is_running(self) -> bool:
        return self.current is not None

    def request(self, name: str, status_text: str, worker: TaskWorker, on_result: TaskCallback, on_error: TaskCallback, on_final: Callable[[], None]):
        def start_task():
            queue: Queue = Queue()
            stop_event = threading.Event()

            def emit(event: str, payload: object):
                queue.put((event, payload))

            def wrapper():
                try:
                    worker(stop_event, emit)
                except Exception as exc:  # noqa: BLE001
                    queue.put(("error", exc))
                finally:
                    queue.put(("done", None))

            thread = threading.Thread(target=wrapper, daemon=True)
            self.current = _Task(
                name=name,
                thread=thread,
                queue=queue,
                stop_event=stop_event,
                on_result=on_result,
                on_error=on_error,
                on_finished=on_final,
            )
            self.status_callback(status_text)
            thread.start()
            self.root.after(self.poll_interval, self._poll_current)

        if self.current is not None:
            self.log_callback(f"正在取消 {self.current.name} ...")
            self.current.stop_event.set()
            self.pending = start_task
        else:
            start_task()

    def cancel(self):
        if self.current:
            self.current.stop_event.set()

    def _poll_current(self):
        task = self.current
        if not task:
            if self.pending:
                pending = self.pending
                self.pending = None
                pending()
            return

        try:
            while True:
                event, payload = task.queue.get_nowait()
                if event == "log":
                    self.log_callback(str(payload))
                elif event == "result":
                    task.on_result(payload)
                elif event == "error":
                    task.on_error(payload)
                elif event == "done":
                    task.on_finished()
                    self.current = None
                    if self.pending:
                        pending = self.pending
                        self.pending = None
                        pending()
                    return
        except Empty:
            self.root.after(self.poll_interval, self._poll_current)
