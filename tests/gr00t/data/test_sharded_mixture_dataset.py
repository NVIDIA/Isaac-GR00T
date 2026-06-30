from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics


def test_merge_statistics_skips_relative_stats_sidecar_only_input():
    stats = [{"__fingerprints__": {}}]

    merged = merge_statistics(stats, [1.0], is_relative_stats=True)

    assert merged == {}
