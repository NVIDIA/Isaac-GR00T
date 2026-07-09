#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import structlog


class ContextTimer:

    def __init__(self, trainer):
        self.last_key = None
        self.trainer = trainer
        self.start_times = {}
        self.key_stack = []

    def with_label(self, key):
        self.last_key = key
        return self

    def __enter__(self):
        self.key_stack.append(self.last_key)  # Push key to stack
        self.start_times[self.last_key] = time.time()  # Start timing for this key
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        key = self.key_stack.pop()  # Pop key from stack
        diff = time.time() - self.start_times[key]
        self.trainer.log({f"{key}_time": diff})
        structlog.get_logger().info("training_timer", label=key, duration_s=round(diff, 4))


if __name__ == "__main__":

    class MockTrainer:
        def log(self, data):
            print("Logging:", data)

    trainer = MockTrainer()
    my_timer = ContextTimer(trainer)

    with my_timer.with_label("outer"):
        time.sleep(1)
        with my_timer.with_label("inner"):
            time.sleep(2)
        with my_timer.with_label("inner"):  # Another inner block
            time.sleep(1)
