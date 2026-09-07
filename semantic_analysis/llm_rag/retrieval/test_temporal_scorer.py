from semantic_analysis.llm_rag.query_parser import parse_query
from semantic_analysis.llm_rag.retrieval.scorer import (
    action_event_score,
    object_match_score,
)
from semantic_analysis.llm_rag.retrieval.search import search


def _segment(idx, start, labels):
    return {
        "video_id": "demo_cctv",
        "segment_id": f"demo_cctv_seg_{idx:02d}",
        "start_ts": start,
        "end_ts": start + 1,
        "objects": [{"label": label, "confidence": 0.9} for label in labels],
        "ocr": [],
    }


def test_vehicle_query_matches_car_detector_label():
    score = object_match_score(
        [{"object": "vehicle", "attributes": []}],
        [{"label": "car", "confidence": 0.92}],
    )

    assert score == 1.0


def test_departure_prefers_last_visible_vehicle_segment():
    segments = [
        _segment(0, 0, ["car"]),
        _segment(1, 5, ["car", "person"]),
        _segment(2, 10, ["person"]),
    ]
    query = parse_query("when did the car leave").model_dump()

    scores = [action_event_score(query, seg, segments) for seg in segments]

    assert scores[1] == 1.0
    assert scores[1] > scores[0]


def test_arrival_prefers_first_visible_vehicle_segment():
    segments = [
        _segment(0, 0, ["car"]),
        _segment(1, 5, ["car", "person"]),
        _segment(2, 10, ["person"]),
    ]
    query = parse_query("when did the car arrive").model_dump()

    scores = [action_event_score(query, seg, segments) for seg in segments]

    assert scores[0] == 1.0
    assert scores[0] > scores[1]


def test_thieves_getting_out_prefers_person_vehicle_overlap():
    segments = [
        _segment(0, 0, ["car"]),
        _segment(1, 5, ["car", "person"]),
        _segment(2, 10, ["person"]),
    ]
    query = parse_query("when did the thieves get out of the car").model_dump()

    scores = [action_event_score(query, seg, segments) for seg in segments]

    assert scores[1] == 0.95
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


class _FakeEmbedder:
    def embed_visual_query(self, raw_query):
        return [0.0]

    def embed_audio_query(self, raw_query):
        return [0.0]


class _FakeIndex:
    def search_visual(self, query_vec, top_k):
        return []

    def search_audio(self, query_vec, top_k):
        return []


def test_search_ranks_departure_segment_before_first_car_sighting():
    segments = [
        _segment(0, 0, ["car"]),
        _segment(1, 5, ["car", "person"]),
        _segment(2, 10, ["person"]),
    ]
    query = parse_query("when did the car leave").model_dump()

    results = search(
        structured_query=query,
        raw_query="when did the car leave",
        unified_segments=segments,
        embedder=_FakeEmbedder(),
        index_mgr=_FakeIndex(),
    )

    assert results[0]["segment_id"] == "demo_cctv_seg_01"
    assert results[0]["scores"]["event_match"] == 1.0
