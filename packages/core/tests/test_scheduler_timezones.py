from datetime import UTC, datetime

import pytest

from harness.core.scheduler_models import ScheduleSpec
from harness.core.scheduler_runtime import compute_next_run_at, parse_schedule_spec


def test_old_schedule_defaults_to_utc_and_timezone_roundtrips():
    assert ScheduleSpec.from_dict({"kind": "cron", "value": "0 9 * * *"}).timezone == "UTC"
    schedule = parse_schedule_spec(cron="0 9 * * *", timezone="Europe/Tallinn")
    assert ScheduleSpec.from_dict(schedule.to_dict()) == schedule
    assert (
        compute_next_run_at(schedule=schedule, now=datetime(2026, 7, 1, tzinfo=UTC))
        == "2026-07-01T06:00:00+00:00"
    )


def test_spring_gap_is_skipped_and_autumn_folds_are_distinct_instants():
    schedule = parse_schedule_spec(cron="30 3 * * *", timezone="Europe/Tallinn")
    assert (
        compute_next_run_at(schedule=schedule, now=datetime(2026, 3, 29, tzinfo=UTC))
        == "2026-03-30T00:30:00+00:00"
    )
    assert (
        compute_next_run_at(schedule=schedule, now=datetime(2026, 10, 25, 0, 31, tzinfo=UTC))
        == "2026-10-25T01:30:00+00:00"
    )


@pytest.mark.parametrize("at", ["2026-03-29T03:30:00", "2026-10-25T03:30:00"])
def test_ambiguous_or_missing_one_shot_requires_offset(at):
    with pytest.raises(ValueError, match="explicit UTC offset"):
        parse_schedule_spec(at=at, timezone="Europe/Tallinn")


def test_cron_validates_all_fields_and_supports_ranges_steps_and_day_or_weekday():
    with pytest.raises(ValueError):
        parse_schedule_spec(cron="0 9 * 99 *")
    schedule = parse_schedule_spec(cron="0-30/15 9 1 * 1")
    assert (
        compute_next_run_at(schedule=schedule, now=datetime(2026, 9, 7, 9, 1, tzinfo=UTC))
        == "2026-09-07T09:15:00+00:00"
    )
