from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import requests

from qa.retrieval_state import QueryPlan
from qa.retrieval_utils import is_textual_content_type, normalize_doi


_RETRYABLE_STATUS_CODES = {408, 429}


@dataclass
class FetchedDocument:
    url: str
    content_type: str
    text: Optional[str] = None
    binary: Optional[bytes] = None


@dataclass
class ProviderHealthRecord:
    status: str = "idle"
    calls: int = 0
    successes: int = 0
    retry_exhausted_failures: int = 0
    skipped_calls: int = 0
    last_error: Optional[str] = None


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        *,
        provider: str,
        failure_kind: str,
        message: str,
        retry_exhausted: bool = False,
        recoverable: bool = False,
        provider_unavailable: bool = False,
        status_code: Optional[int] = None,
        attempts: int = 1,
    ) -> None:
        self.provider = provider
        self.failure_kind = failure_kind
        self.retry_exhausted = bool(retry_exhausted)
        self.recoverable = bool(recoverable)
        self.provider_unavailable = bool(provider_unavailable)
        self.status_code = status_code
        self.attempts = max(1, int(attempts))
        super().__init__(message)


class ProviderUnavailableError(RuntimeError):
    def __init__(self, *, provider: str, last_error: Optional[str] = None) -> None:
        self.provider = provider
        self.last_error = str(last_error or "").strip() or None
        message = f"{provider} provider unavailable; subsequent calls skipped for this run."
        if self.last_error:
            message = f"{message} Last error: {self.last_error}"
        super().__init__(message)


class _RetryableHttpStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = int(status_code)
        super().__init__(message)


class _HttpTransportMixin:
    def _init_transport(
        self,
        *,
        provider_name: str,
        timeout: float,
        retry_attempts: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        disable_on_retry_exhausted: bool = True,
        request_get: Optional[Callable[..., Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.provider_name = provider_name
        self.timeout = float(timeout)
        self.retry_attempts = max(0, int(retry_attempts))
        self.backoff_base_seconds = max(0.0, float(backoff_base_seconds))
        self.backoff_max_seconds = max(self.backoff_base_seconds, float(backoff_max_seconds))
        self.disable_on_retry_exhausted = bool(disable_on_retry_exhausted)
        self._request_get = request_get or requests.get
        self._sleep_fn = sleep_fn or time.sleep
        self._random_fn = random_fn or random.random
        self._health = ProviderHealthRecord()

    def health_snapshot(self) -> Dict[str, Any]:
        return asdict(self._health)

    def is_available(self) -> bool:
        return self._health.status != "unavailable"

    def _perform_get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if not self.is_available():
            self._health.skipped_calls += 1
            raise ProviderUnavailableError(
                provider=self.provider_name,
                last_error=self._health.last_error,
            )

        self._health.calls += 1
        total_attempts = self.retry_attempts + 1
        last_error = ""

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._request_get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                status_code = int(getattr(response, "status_code", 200))
                if self._should_retry_status(status_code):
                    raise _RetryableHttpStatusError(
                        status_code,
                        f"HTTP {status_code} from {self.provider_name}",
                    )
                response.raise_for_status()
                self._health.successes += 1
                self._health.status = "healthy"
                self._health.last_error = None
                return response
            except requests.exceptions.Timeout as exc:
                last_error = f"{self.provider_name} timeout: {exc}"
                if attempt < total_attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise self._finalize_recoverable_error(
                    failure_kind="timeout",
                    message=last_error,
                    attempts=attempt,
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = f"{self.provider_name} connection error: {exc}"
                if attempt < total_attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise self._finalize_recoverable_error(
                    failure_kind="failure",
                    message=last_error,
                    attempts=attempt,
                ) from exc
            except _RetryableHttpStatusError as exc:
                last_error = str(exc)
                failure_kind = "timeout" if exc.status_code == 408 else "failure"
                if attempt < total_attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise self._finalize_recoverable_error(
                    failure_kind=failure_kind,
                    message=last_error,
                    attempts=attempt,
                    status_code=exc.status_code,
                ) from exc
            except requests.exceptions.HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                last_error = f"{self.provider_name} HTTP error{f' {status_code}' if status_code else ''}: {exc}"
                failure_kind = "timeout" if status_code == 408 else "failure"
                recoverable = self._should_retry_status(status_code)
                if recoverable and attempt < total_attempts:
                    self._sleep_backoff(attempt)
                    continue
                if recoverable:
                    raise self._finalize_recoverable_error(
                        failure_kind=failure_kind,
                        message=last_error,
                        attempts=attempt,
                        status_code=status_code,
                    ) from exc
                raise self._finalize_nonrecoverable_error(
                    failure_kind=failure_kind,
                    message=last_error,
                    attempts=attempt,
                    status_code=status_code,
                ) from exc
            except requests.exceptions.RequestException as exc:
                last_error = f"{self.provider_name} request error: {exc}"
                if attempt < total_attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise self._finalize_recoverable_error(
                    failure_kind="failure",
                    message=last_error,
                    attempts=attempt,
                ) from exc

        raise self._finalize_nonrecoverable_error(
            failure_kind="failure",
            message=last_error or f"{self.provider_name} request failed",
            attempts=total_attempts,
        )

    def _finalize_recoverable_error(
        self,
        *,
        failure_kind: str,
        message: str,
        attempts: int,
        status_code: Optional[int] = None,
    ) -> ProviderRequestError:
        self._health.retry_exhausted_failures += 1
        retry_prefix = f"retry exhausted after {attempts} attempt{'s' if attempts != 1 else ''}"
        self._health.last_error = f"{retry_prefix}: {message}"
        self._health.status = "unavailable" if self.disable_on_retry_exhausted else "degraded"
        return ProviderRequestError(
            provider=self.provider_name,
            failure_kind=failure_kind,
            message=(
                f"{self._health.last_error}"
                + ("; provider marked unavailable for this run" if self.disable_on_retry_exhausted else "")
            ),
            retry_exhausted=True,
            recoverable=True,
            provider_unavailable=self.disable_on_retry_exhausted,
            status_code=status_code,
            attempts=attempts,
        )

    def _finalize_nonrecoverable_error(
        self,
        *,
        failure_kind: str,
        message: str,
        attempts: int,
        status_code: Optional[int] = None,
    ) -> ProviderRequestError:
        self._health.last_error = message
        if self._health.status == "idle":
            self._health.status = "degraded"
        return ProviderRequestError(
            provider=self.provider_name,
            failure_kind=failure_kind,
            message=message,
            retry_exhausted=False,
            recoverable=False,
            provider_unavailable=False,
            status_code=status_code,
            attempts=attempts,
        )

    def _sleep_backoff(self, attempt_index: int) -> None:
        delay = min(
            self.backoff_max_seconds,
            self.backoff_base_seconds * (2 ** max(0, attempt_index - 1)),
        )
        if delay <= 0:
            return
        jitter = min(delay * 0.1, 0.25) * max(0.0, float(self._random_fn()))
        self._sleep_fn(delay + jitter)

    def _should_retry_status(self, status_code: Optional[int]) -> bool:
        if status_code is None:
            return False
        return int(status_code) in _RETRYABLE_STATUS_CODES or int(status_code) >= 500


class OpenAlexClient(_HttpTransportMixin):
    def __init__(
        self,
        base_url: str = "https://api.openalex.org/works",
        timeout: float = 10.0,
        mailto: Optional[str] = None,
        retry_attempts: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        request_get: Optional[Callable[..., Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.base_url = base_url
        self.mailto = mailto
        self._init_transport(
            provider_name="openalex",
            timeout=timeout,
            retry_attempts=retry_attempts,
            backoff_base_seconds=backoff_base_seconds,
            backoff_max_seconds=backoff_max_seconds,
            disable_on_retry_exhausted=True,
            request_get=request_get,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )

    def search(self, query_plan: QueryPlan, limit: int = 8) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "search": query_plan.query_text,
            "per-page": limit,
        }
        filters: List[str] = []
        if query_plan.year_from is not None or query_plan.year_to is not None:
            year_from = query_plan.year_from or query_plan.year_to
            year_to = query_plan.year_to or query_plan.year_from
            if year_from is not None and year_to is not None:
                filters.append(f"publication_year:{year_from}-{year_to}")
        if filters:
            params["filter"] = ",".join(filters)
        if self.mailto:
            params["mailto"] = self.mailto
        response = self._perform_get(self.base_url, params=params)
        payload = response.json()
        return list(payload.get("results") or [])


class CrossrefClient(_HttpTransportMixin):
    def __init__(
        self,
        base_url: str = "https://api.crossref.org/works",
        timeout: float = 10.0,
        mailto: Optional[str] = None,
        retry_attempts: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        request_get: Optional[Callable[..., Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mailto = mailto
        self._init_transport(
            provider_name="crossref",
            timeout=timeout,
            retry_attempts=retry_attempts,
            backoff_base_seconds=backoff_base_seconds,
            backoff_max_seconds=backoff_max_seconds,
            disable_on_retry_exhausted=True,
            request_get=request_get,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )

    def enrich(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doi = normalize_doi(candidate.get("doi"))
        if doi:
            response = self._perform_get(
                f"{self.base_url}/{quote(doi, safe='')}",
                params={"mailto": self.mailto} if self.mailto else None,
            )
            return response.json().get("message")

        params: Dict[str, Any] = {"rows": 1, "query.title": candidate.get("title")}
        year = candidate.get("year")
        if year:
            params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year + 1}-12-31"
        if self.mailto:
            params["mailto"] = self.mailto
        response = self._perform_get(self.base_url, params=params)
        items = response.json().get("message", {}).get("items") or []
        return items[0] if items else None


class SemanticScholarClient(_HttpTransportMixin):
    def __init__(
        self,
        base_url: str = "https://api.semanticscholar.org/graph/v1/paper/search",
        timeout: float = 10.0,
        api_key: Optional[str] = None,
        retry_attempts: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        request_get: Optional[Callable[..., Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._init_transport(
            provider_name="semantic_scholar",
            timeout=timeout,
            retry_attempts=retry_attempts,
            backoff_base_seconds=backoff_base_seconds,
            backoff_max_seconds=backoff_max_seconds,
            disable_on_retry_exhausted=True,
            request_get=request_get,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )

    def enrich(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        params = {
            "query": candidate.get("title"),
            "limit": 1,
            "fields": "title,abstract,citationCount,year,venue,authors,externalIds",
        }
        headers = {"x-api-key": self.api_key} if self.api_key else None
        response = self._perform_get(self.base_url, params=params, headers=headers)
        data = response.json().get("data") or []
        return data[0] if data else None


class UnpaywallClient(_HttpTransportMixin):
    def __init__(
        self,
        email: Optional[str],
        base_url: str = "https://api.unpaywall.org/v2",
        timeout: float = 10.0,
        retry_attempts: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        request_get: Optional[Callable[..., Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.email = email
        self.base_url = base_url.rstrip("/")
        self._init_transport(
            provider_name="unpaywall",
            timeout=timeout,
            retry_attempts=retry_attempts,
            backoff_base_seconds=backoff_base_seconds,
            backoff_max_seconds=backoff_max_seconds,
            disable_on_retry_exhausted=True,
            request_get=request_get,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )

    def lookup(self, doi: Optional[str]) -> Optional[Dict[str, Any]]:
        normalized_doi = normalize_doi(doi)
        if not normalized_doi or not self.email:
            return None
        response = self._perform_get(
            f"{self.base_url}/{quote(normalized_doi, safe='')}",
            params={"email": self.email},
        )
        return response.json()


class HttpTextFetcher(_HttpTransportMixin):
    def __init__(
        self,
        timeout: float = 15.0,
        retry_attempts: int = 2,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 8.0,
        request_get: Optional[Callable[..., Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        random_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self._init_transport(
            provider_name="oa_fetch",
            timeout=timeout,
            retry_attempts=retry_attempts,
            backoff_base_seconds=backoff_base_seconds,
            backoff_max_seconds=backoff_max_seconds,
            disable_on_retry_exhausted=False,
            request_get=request_get,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )

    def fetch(self, url: str) -> FetchedDocument:
        response = self._perform_get(
            url,
            headers={"User-Agent": "ChemQA/1.0 (+https://github.com/openai/codex-cli)"},
        )
        content_type = str(response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type == "application/pdf" or url.lower().endswith(".pdf"):
            return FetchedDocument(url=url, content_type="application/pdf", binary=response.content)
        if is_textual_content_type(content_type):
            return FetchedDocument(url=url, content_type=content_type or "text/plain", text=response.text)
        return FetchedDocument(
            url=url,
            content_type=content_type or "application/octet-stream",
            binary=response.content,
        )
