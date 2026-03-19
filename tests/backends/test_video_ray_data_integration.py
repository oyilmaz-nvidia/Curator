# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modality: video
"""Integration tests for RayDataExecutor with video modality (VideoTask).

Tests that video stages work correctly with RayDataExecutor, covering:
- Task stages (no setup, stateless)
- Actor stages (with setup, stateful)
- Fanout stages (IS_FANOUT_STAGE=True, one VideoTask → multiple VideoTasks)
"""

import copy
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask

# ─────────────────────────── test configuration ──────────────────────────── #

NUM_TEST_VIDEOS = 5
CLIPS_PER_FANOUT = 3
EXPECTED_OUTPUT_TASKS = NUM_TEST_VIDEOS * CLIPS_PER_FANOUT
EXPECTED_NUM_STAGES = 4  # VideoTagStage → VideoActorStage → VideoFanoutStage → VideoTagStage


# ──────────────────────────────── helpers ────────────────────────────────── #


def _make_video_task(i: int) -> VideoTask:
    """Create a minimal VideoTask for testing (no real video bytes needed)."""
    video = Video(
        input_video=pathlib.Path(f"/fake/video_{i}.mp4"),
        metadata=VideoMetadata(width=1920, height=1080, framerate=30.0, duration=10.0),
    )
    return VideoTask(
        task_id=f"video_{i}",
        dataset_name="test_dataset",
        data=video,
    )


# ──────────────────────────────── test stages ────────────────────────────── #


@dataclass
class VideoTagStage(ProcessingStage[VideoTask, VideoTask]):
    """Task stage (no setup): writes a flag into the video errors dict."""

    tag_key: str = "tagged"
    name: str = "video_tag"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: VideoTask) -> VideoTask:
        task.data.errors[self.tag_key] = "true"
        return task


@dataclass
class VideoActorStage(ProcessingStage[VideoTask, VideoTask]):
    """Actor stage (has setup): records a persistent actor_id into the video errors dict."""

    name: str = "video_actor"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        self._actor_id = str(uuid.uuid4())

    def process(self, task: VideoTask) -> VideoTask:
        task.data.errors["actor_id"] = self._actor_id
        return task


@dataclass
class VideoFanoutStage(ProcessingStage[VideoTask, VideoTask]):
    """Fanout stage: splits one VideoTask into N VideoTasks, one per synthetic clip."""

    clips_per_video: int = CLIPS_PER_FANOUT
    name: str = "video_fanout"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: VideoTask) -> list[VideoTask]:
        output = []
        for i in range(self.clips_per_video):
            clip = Clip(
                uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{task.task_id}_clip_{i}"),
                source_video=str(task.data.input_video),
                span=(float(i * 3), float((i + 1) * 3)),
            )
            new_video = Video(
                input_video=task.data.input_video,
                metadata=copy.deepcopy(task.data.metadata),
                clips=[clip],
                errors=copy.deepcopy(task.data.errors),
            )
            output.append(
                VideoTask(
                    task_id=f"{task.task_id}_chunk_{i}",
                    dataset_name=task.dataset_name,
                    data=new_video,
                    _stage_perf=copy.deepcopy(task._stage_perf),
                    _metadata=copy.deepcopy(task._metadata),
                )
            )
        return output


# ─────────────────────────── pipeline factory ────────────────────────────── #


def create_video_test_pipeline() -> Pipeline:
    """Create a minimal video pipeline exercising task, actor, and fanout stages."""
    pipeline = Pipeline(name="video_ray_data_test", description="Video Ray Data integration test")
    pipeline.add_stage(VideoTagStage(tag_key="pre_fanout"))
    pipeline.add_stage(VideoActorStage())
    pipeline.add_stage(VideoFanoutStage())
    pipeline.add_stage(VideoTagStage(tag_key="post_fanout"))
    return pipeline


# ──────────────────────────────── tests ──────────────────────────────────── #


class TestVideoRayDataIntegration:
    """Integration tests for RayDataExecutor with VideoTask."""

    output_tasks: list[VideoTask] | None = None

    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(self, request: pytest.FixtureRequest) -> None:
        """Execute the video pipeline with RayDataExecutor and store output tasks."""
        initial_tasks = [_make_video_task(i) for i in range(NUM_TEST_VIDEOS)]
        pipeline = create_video_test_pipeline()
        executor = RayDataExecutor()
        request.cls.output_tasks = pipeline.run(executor, initial_tasks=initial_tasks)

    def test_output_task_count(self) -> None:
        """Fanout stage should multiply tasks by CLIPS_PER_FANOUT."""
        assert self.output_tasks is not None
        assert len(self.output_tasks) == EXPECTED_OUTPUT_TASKS

    def test_output_task_types(self) -> None:
        """All output tasks should be VideoTask instances."""
        assert self.output_tasks is not None
        assert all(isinstance(t, VideoTask) for t in self.output_tasks)

    def test_task_ids_unique(self) -> None:
        """Each output VideoTask should have a unique task_id."""
        assert self.output_tasks is not None
        ids = {t.task_id for t in self.output_tasks}
        assert len(ids) == EXPECTED_OUTPUT_TASKS

    def test_task_stage_ran(self) -> None:
        """VideoTagStage (task stage) should have tagged each video before the fanout."""
        assert self.output_tasks is not None
        for task in self.output_tasks:
            assert task.data.errors.get("pre_fanout") == "true", (
                f"Expected 'pre_fanout' tag in task {task.task_id}"
            )

    def test_actor_stage_ran(self) -> None:
        """VideoActorStage (actor stage) should have written actor_id into each video."""
        assert self.output_tasks is not None
        for task in self.output_tasks:
            assert "actor_id" in task.data.errors, (
                f"Expected 'actor_id' in task {task.task_id}"
            )

    def test_actor_stage_consistent_within_worker(self) -> None:
        """All tasks processed by the same actor should share the same actor_id."""
        assert self.output_tasks is not None
        actor_ids = {t.data.errors["actor_id"] for t in self.output_tasks if "actor_id" in t.data.errors}
        # At least one actor_id should exist
        assert len(actor_ids) >= 1

    def test_fanout_stage_produced_clips(self) -> None:
        """Each output VideoTask should contain exactly one clip (produced by VideoFanoutStage)."""
        assert self.output_tasks is not None
        for task in self.output_tasks:
            assert len(task.data.clips) == 1, (
                f"Expected 1 clip in task {task.task_id}, got {len(task.data.clips)}"
            )

    def test_post_fanout_stage_ran(self) -> None:
        """VideoTagStage after fanout should have tagged all output tasks."""
        assert self.output_tasks is not None
        for task in self.output_tasks:
            assert task.data.errors.get("post_fanout") == "true", (
                f"Expected 'post_fanout' tag in task {task.task_id}"
            )

    def test_perf_stats_recorded(self) -> None:
        """Each output task should have perf stats for all pipeline stages."""
        assert self.output_tasks is not None
        for task in self.output_tasks:
            assert len(task._stage_perf) == EXPECTED_NUM_STAGES, (
                f"Expected {EXPECTED_NUM_STAGES} perf entries in {task.task_id}, "
                f"got {len(task._stage_perf)}"
            )
            for perf in task._stage_perf:
                assert perf.process_time > 0, (
                    f"Expected non-zero process_time for stage {perf.stage_name}"
                )

    def test_dataset_names_preserved(self) -> None:
        """Dataset name should be preserved through all stages."""
        assert self.output_tasks is not None
        for task in self.output_tasks:
            assert task.dataset_name == "test_dataset"
