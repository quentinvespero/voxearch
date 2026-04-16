import pytest

from src.utils import normalize_url, is_hf_model_cached


# ── normalize_url ─────────────────────────────────────────────────────────────

class TestNormalizeUrl:
    def test_strips_youtube_si_param(self):
        url = "https://www.youtube.com/watch?v=abc123&si=TRACKINGTOKEN"
        assert normalize_url(url) == "https://www.youtube.com/watch?v=abc123"

    def test_strips_utm_params(self):
        url = "https://example.com/ep1?utm_source=newsletter&utm_medium=email"
        assert normalize_url(url) == "https://example.com/ep1"

    def test_preserves_youtube_v_param(self):
        url = "https://www.youtube.com/watch?v=abc123"
        assert normalize_url(url) == "https://www.youtube.com/watch?v=abc123"

    def test_preserves_youtube_list_param(self):
        url = "https://www.youtube.com/watch?v=abc123&list=PLxxx"
        result = normalize_url(url)
        assert "v=abc123" in result
        assert "list=PLxxx" in result

    def test_preserves_youtube_t_param(self):
        url = "https://www.youtube.com/watch?v=abc123&t=30s"
        result = normalize_url(url)
        assert "v=abc123" in result
        assert "t=30s" in result

    def test_strips_si_preserves_content_params(self):
        url = "https://www.youtube.com/watch?v=abc123&si=TOKEN&list=PLxxx"
        result = normalize_url(url)
        assert "si=" not in result
        assert "v=abc123" in result
        assert "list=PLxxx" in result

    def test_strips_feature_param(self):
        url = "https://www.youtube.com/watch?v=abc123&feature=shared"
        assert normalize_url(url) == "https://www.youtube.com/watch?v=abc123"

    def test_strips_pp_param(self):
        url = "https://www.youtube.com/watch?v=abc123&pp=SHORTS_TOKEN"
        assert normalize_url(url) == "https://www.youtube.com/watch?v=abc123"

    def test_no_params_unchanged(self):
        url = "https://example.com/episode"
        assert normalize_url(url) == "https://example.com/episode"

    def test_params_sorted_for_stability(self):
        # Same params in different order → same canonical URL
        url_a = "https://example.com/?z=1&a=2"
        url_b = "https://example.com/?a=2&z=1"
        assert normalize_url(url_a) == normalize_url(url_b)

    def test_all_utm_variants_stripped(self):
        url = (
            "https://example.com/ep"
            "?utm_source=x&utm_medium=y&utm_campaign=z"
            "&utm_term=t&utm_content=c"
        )
        assert normalize_url(url) == "https://example.com/ep"


# ── is_hf_model_cached ────────────────────────────────────────────────────────

class TestIsHfModelCached:
    def test_returns_false_when_cache_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        # tmp_path/hub/ doesn't exist → no models cached
        assert is_hf_model_cached("some-org/some-model") is False

    def test_returns_true_for_exact_repo_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        hub = tmp_path / "hub"
        (hub / "models--some-org--some-model").mkdir(parents=True)
        assert is_hf_model_cached("some-org/some-model") is True

    def test_returns_true_for_sentence_transformers_prefix(self, tmp_path, monkeypatch):
        # When the bare repo name has no org, it also checks the sentence-transformers/ prefix
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        hub = tmp_path / "hub"
        (hub / "models--sentence-transformers--my-model").mkdir(parents=True)
        assert is_hf_model_cached("my-model") is True

    def test_returns_false_when_wrong_model_cached(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        hub = tmp_path / "hub"
        (hub / "models--other-org--other-model").mkdir(parents=True)
        assert is_hf_model_cached("some-org/some-model") is False
