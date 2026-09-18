from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

settings = get_settings()

_bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def get_jwk_client() -> jwt.PyJWKClient:
    """
    Cached JWKS client for verifying Supabase session tokens.

    This project signs auth tokens with ES256 (asymmetric keys), confirmed by
    fetching {SUPABASE_URL}/auth/v1/.well-known/jwks.json directly rather than
    assuming the legacy shared-secret (HS256) scheme older Supabase projects
    use. PyJWKClient fetches and caches the public signing keys, so this
    keeps working across key rotation without redeploying.
    """
    return jwt.PyJWKClient(f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json")


def get_current_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """
    Verify the Supabase-issued bearer token and return the caller's user id
    (the token's `sub` claim). Identity always comes from the verified token,
    never from a request body or path parameter.

    Deliberately sync, not async: get_jwk_client() can make a blocking HTTP
    call on a cache miss (first request, or after key rotation), and jwt.decode
    itself is blocking CPU work. As `async def` this would stall the event
    loop with no `await` point to yield at; FastAPI runs a sync dependency
    like this in the threadpool instead.
    """
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        signing_key = get_jwk_client().get_signing_key_from_jwt(creds.credentials)
        payload = jwt.decode(
            creds.credentials,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token has no subject")
    return sub
