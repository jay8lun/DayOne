"""领域异常：把内部错误翻译成可读的 HTTP 响应，避免 500 裸奔。"""

from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class AdapterError(AppError):
    status_code = 502
    code = "adapter_error"


class AdapterNotConfigured(AppError):
    status_code = 400
    code = "adapter_not_configured"


class KnowledgeBaseEmpty(AppError):
    status_code = 409
    code = "knowledge_base_empty"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"
