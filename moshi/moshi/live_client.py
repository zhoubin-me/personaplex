# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import asyncio
from dataclasses import dataclass, field
import ssl
import sys
import threading
from typing import Optional
from urllib.parse import urlencode

import aiohttp
import numpy as np
import sounddevice as sd
import sphn


DEFAULT_SAMPLE_RATE = 24000
DEFAULT_BLOCK_MS = 20


@dataclass
class PlaybackBuffer:
    data: np.ndarray

    def __init__(self):
        self.data = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray):
        if chunk.ndim != 1:
            chunk = np.asarray(chunk).reshape(-1)
        with self._lock:
            self.data = np.concatenate([self.data, chunk.astype(np.float32, copy=False)])

    def pop(self, size: int) -> np.ndarray:
        with self._lock:
            if self.data.shape[0] >= size:
                out = self.data[:size]
                self.data = self.data[size:]
            else:
                out = np.zeros(size, dtype=np.float32)
                available = self.data.shape[0]
                if available > 0:
                    out[:available] = self.data
                    self.data = np.zeros(0, dtype=np.float32)
        return out


@dataclass
class Stats:
    mic_chunks: int = 0
    mic_rms_accum: float = 0.0
    tx_packets: int = 0
    tx_bytes: int = 0
    rx_packets: int = 0
    rx_bytes: int = 0
    play_samples: int = 0
    text_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_mic_chunk(self, chunk: np.ndarray):
        with self._lock:
            self.mic_chunks += 1
            self.mic_rms_accum += float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))

    def add_tx(self, nbytes: int):
        with self._lock:
            self.tx_packets += 1
            self.tx_bytes += nbytes

    def add_rx(self, nbytes: int):
        with self._lock:
            self.rx_packets += 1
            self.rx_bytes += nbytes

    def add_play(self, nsamples: int):
        with self._lock:
            self.play_samples += nsamples

    def add_text(self):
        with self._lock:
            self.text_tokens += 1

    def snapshot_and_reset(self):
        with self._lock:
            snap = (
                self.mic_chunks,
                self.mic_rms_accum,
                self.tx_packets,
                self.tx_bytes,
                self.rx_packets,
                self.rx_bytes,
                self.play_samples,
                self.text_tokens,
            )
            self.mic_chunks = 0
            self.mic_rms_accum = 0.0
            self.tx_packets = 0
            self.tx_bytes = 0
            self.rx_packets = 0
            self.rx_bytes = 0
            self.play_samples = 0
            self.text_tokens = 0
        return snap


def _build_chat_url(
    base_url: str,
    voice_prompt: str,
    text_prompt: str,
    seed: Optional[int],
) -> str:
    params = {
        "voice_prompt": voice_prompt,
        "text_prompt": text_prompt,
    }
    if seed is not None:
        params["seed"] = str(seed)
    return f"{base_url}?{urlencode(params)}"


def _create_ssl_context(insecure: bool) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def run_client(
    url: str,
    sample_rate: int,
    block_ms: int,
    insecure: bool,
    input_device: Optional[str | int],
    output_device: Optional[str | int],
):
    block_size = int(sample_rate * block_ms / 1000.0)
    opus_writer = sphn.OpusStreamWriter(sample_rate)
    opus_reader = sphn.OpusStreamReader(sample_rate)

    mic_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
    text_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4096)
    playback = PlaybackBuffer()
    stop_event = asyncio.Event()
    started = asyncio.Event()
    stats = Stats()
    first_tx_printed = False
    first_rx_printed = False
    loop = asyncio.get_running_loop()

    def _push_mic_chunk(chunk: np.ndarray):
        if not started.is_set():
            # Keep capture live but avoid queueing stale pre-handshake audio.
            return
        if mic_queue.full():
            try:
                _ = mic_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            mic_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    def mic_callback(indata, _frames, _time, status):
        if status:
            print(f"[mic] {status}")
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        stats.add_mic_chunk(mono)
        loop.call_soon_threadsafe(_push_mic_chunk, mono)

    def out_callback(outdata, frames, _time, status):
        if status:
            print(f"[spk] {status}")
        out_chunk = playback.pop(frames)
        outdata[:, 0] = out_chunk
        stats.add_play(frames)

    ssl_ctx = _create_ssl_context(insecure) if url.startswith("wss://") else None

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, ssl=ssl_ctx) as ws:
            print(f"Connected to {url}, waiting for handshake...")

            async def recv_loop():
                nonlocal first_rx_printed
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.BINARY:
                        continue
                    data = msg.data
                    if not data:
                        continue
                    kind = data[0]
                    payload = data[1:]
                    if kind == 0:
                        print("Handshake received. Streaming started.")
                        # Drop any stale chunks captured before handshake.
                        while True:
                            try:
                                _ = mic_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        started.set()
                    elif kind == 1:
                        stats.add_rx(len(payload))
                        if not first_rx_printed:
                            print("\n[debug] first audio packet received from server", flush=True)
                            first_rx_printed = True
                        opus_reader.append_bytes(payload)
                        pcm = opus_reader.read_pcm()
                        if pcm.shape[-1] > 0:
                            playback.append(pcm.astype(np.float32, copy=False))
                    elif kind == 2:
                        stats.add_text()
                        text = payload.decode("utf-8")
                        if text_queue.full():
                            try:
                                _ = text_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                        try:
                            text_queue.put_nowait(text)
                        except asyncio.QueueFull:
                            pass
                stop_event.set()

            async def text_loop():
                # Batch tiny text tokens so terminal I/O does not block audio handling.
                while not stop_event.is_set():
                    try:
                        token = await asyncio.wait_for(text_queue.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                    chunks = [token]
                    while True:
                        try:
                            chunks.append(text_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    sys.stdout.write("".join(chunks))
                    sys.stdout.flush()

            async def send_loop():
                nonlocal first_tx_printed
                while not stop_event.is_set():
                    if not started.is_set():
                        await asyncio.sleep(0.01)
                        continue
                    chunk = await mic_queue.get()
                    # Bound end-to-end latency by preferring freshest mic audio.
                    while mic_queue.qsize() > 2:
                        try:
                            chunk = mic_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    opus_writer.append_pcm(chunk)
                    payload = opus_writer.read_bytes()
                    if payload:
                        stats.add_tx(len(payload))
                        if not first_tx_printed:
                            print("\n[debug] first audio packet sent to server", flush=True)
                            first_tx_printed = True
                        await ws.send_bytes(b"\x01" + payload)

            async def stats_loop():
                while not stop_event.is_set():
                    await asyncio.sleep(1.0)
                    (
                        mic_chunks,
                        mic_rms_accum,
                        tx_packets,
                        tx_bytes,
                        rx_packets,
                        rx_bytes,
                        play_samples,
                        text_tokens,
                    ) = stats.snapshot_and_reset()
                    mic_rms = (mic_rms_accum / mic_chunks) if mic_chunks > 0 else 0.0
                    print(
                        f"\n[stats] mic_chunks={mic_chunks} mic_rms={mic_rms:.5f} "
                        f"tx={tx_packets}/{tx_bytes}B rx={rx_packets}/{rx_bytes}B "
                        f"play_samples={play_samples} text={text_tokens}",
                        flush=True,
                    )

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=mic_callback,
                device=input_device,
            ), sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=out_callback,
                device=output_device,
            ):
                recv_task = asyncio.create_task(recv_loop())
                text_task = asyncio.create_task(text_loop())
                send_task = asyncio.create_task(send_loop())
                stats_task = asyncio.create_task(stats_loop())
                done, pending = await asyncio.wait(
                    [recv_task, text_task, send_task, stats_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                for task in done:
                    _ = task.result()


def main():
    parser = argparse.ArgumentParser(
        description="Realtime PersonaPlex Python client (mic + speakers)."
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--path", type=str, default="/api/chat")
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Use wss:// (recommended when server runs with --ssl).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable certificate verification for self-signed local certs.",
    )
    parser.add_argument(
        "--voice-prompt",
        type=str,
        default="NATF2.pt",
        help="Voice prompt filename available on the server.",
    )
    parser.add_argument("--text-prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--block-ms", type=int, default=DEFAULT_BLOCK_MS)
    parser.add_argument(
        "--input-device",
        type=str,
        default=None,
        help="Input device name or index for microphone.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        help="Output device name or index for speaker/headphones.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional full ws/wss URL (overrides host/port/path).",
    )
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    if args.url is not None:
        base_url = args.url
    else:
        scheme = "wss" if args.secure else "ws"
        path = args.path if args.path.startswith("/") else f"/{args.path}"
        base_url = f"{scheme}://{args.host}:{args.port}{path}"

    url = _build_chat_url(
        base_url=base_url,
        voice_prompt=args.voice_prompt,
        text_prompt=args.text_prompt,
        seed=args.seed,
    )
    try:
        asyncio.run(
            run_client(
                url=url,
                sample_rate=args.sample_rate,
                block_ms=args.block_ms,
                insecure=args.insecure,
                input_device=args.input_device,
                output_device=args.output_device,
            )
        )
    except KeyboardInterrupt:
        print("\nClient stopped.")


if __name__ == "__main__":
    main()
