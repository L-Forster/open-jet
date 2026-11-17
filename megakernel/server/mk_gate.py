"""The request gate: Host (DNS rebinding) and Origin checks, CORS, the API key."""
import hmac
import io
import ipaddress
import json
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler

import mk_limits
from mk_log import log


# ---- the request gate: Host (DNS rebinding) and Origin (other sites' web pages) checks, CORS for listed origins ----
HOST_NAME = r"(?:\[[0-9A-Za-z:._~%-]+\]|[0-9A-Za-z._~%!$&'()*+,;=-]+)"  # (a reg-name or a bracketed IPv6 address)
AUTHORITY_RE = re.compile(rf"({HOST_NAME})(?::([0-9]{{0,5}}))?")
ORIGIN_RE = re.compile(rf"([A-Za-z][A-Za-z0-9+.-]*)://({HOST_NAME})(?::([0-9]{{0,5}}))?/?")
DEFAULT_PORTS = {"http": 80, "https": 443}
CORS_METHODS = "GET, POST, OPTIONS"
CORS_MAX_AGE = 86400  # seconds a browser may keep a preflight's answer (every request is still checked)
HEADER_LIST_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+(?:[ \t]*,[ \t]*[!#$%&'*+.^_`|~0-9A-Za-z-]+)*")


def is_ip(host):
    """host (as in a URL: IPv6 in brackets) is an IP address literal"""
    try:
        ipaddress.ip_address(host[1:-1] if host.startswith("[") and host.endswith("]") else host)
        return True
    except ValueError:
        return False


def parse_origin(origin):
    """(scheme, host, port) of an Origin header / --cors-origin value, lower case, port None for a scheme without a
    default one; None if it is not an origin ("null" -> ("null", "", None): only a listed "null" matches it)"""
    o = origin.strip()
    if o == "null":
        return "null", "", None
    m = ORIGIN_RE.fullmatch(o)
    if not m or (m.group(3) and int(m.group(3)) > 65535):
        return None
    scheme, host = m.group(1).lower(), m.group(2).lower()
    return scheme, host, int(m.group(3)) if m.group(3) else DEFAULT_PORTS.get(scheme)


def origin_key(origin):
    """the origin as browsers serialize it (scheme://host[:port], without the scheme's default port), or None"""
    p = parse_origin(origin)
    if p is None or p[0] == "null":
        return p and "null"
    scheme, host, port = p
    return f"{scheme}://{host}" + (f":{port}" if port is not None and port != DEFAULT_PORTS.get(scheme) else "")


class Gate:
    """Which requests the server answers (main sets HTTPServer.gate; without one, every request).
    Host: localhost, an IP address, the --host name or an --allow-host one (any port), or none (HTTP/1.0). A web page
    that points its own site's name at this machine (DNS rebinding) is refused: it would be same-origin with the server
    and could read the replies, /props and /metrics.
    Origin (browsers send it with every POST and every cross-site request): the server's own (Origin's host and port ==
    Host's, 80 without a port) or an --cors-origin one; pages of any other origin (other sites, another program's port
    on this machine, file: pages) get a 403, including the simple requests (a text/plain POST) a page may send without a
    preflight. Requests without an Origin are not from a web page (curl, the SDKs, agents) and are answered. An
    admitted origin gets CORS headers (GatedHandler).
    API key (--api-key): every request but /health and CORS preflights must carry it (authorized)."""

    def __init__(self, host="127.0.0.1", allow_hosts=(), cors_origins=(), api_key=None):
        self.api_key = api_key.encode() if api_key else None
        names = [h.strip().lower().rstrip(".") for h in allow_hosts if h.strip()]
        self.any_host = "*" in names
        self.hosts = {"localhost"} | {h for h in names if h != "*"}
        host = (host or "").strip().lower().rstrip(".")
        if host and not is_ip(host):
            self.hosts.add(host)
        self.origins = set()
        for o in cors_origins:
            if not o.strip():
                continue
            if o.strip() == "*":
                raise ValueError("--cors-origin / MK_CORS_ORIGINS: list the origins to allow (any origin is not supported)")
            key = origin_key(o)
            if key is None:
                raise ValueError(f"--cors-origin / MK_CORS_ORIGINS: {o.strip()[:100]!r} is not an origin "
                                 f"(scheme://host[:port], e.g. http://localhost:5173)")
            self.origins.add(key)

    def check(self, headers):
        """(None, the origin to send CORS headers to, or None) for a request to answer; (the 403's message, None) for
        one to refuse"""
        hosts, auth = headers.get_all("Host") or [], None
        if len(hosts) > 1:
            return "more than one Host header", None
        if hosts:
            m = AUTHORITY_RE.fullmatch(hosts[0].strip())
            if not m:
                return f"bad Host header {hosts[0].strip()[:100]!r}", None
            name = m.group(1).lower().rstrip(".")
            auth = name, int(m.group(2)) if m.group(2) else None
            if not (self.any_host or is_ip(name) or name in self.hosts):
                return (f"Host {name[:100]} is not allowed (web pages must not reach the server under other names): "
                        f"start the server with --allow-host {name[:100]} (env MK_ALLOW_HOSTS) to accept it"), None
        origins = headers.get_all("Origin") or []
        if not origins:
            return None, None
        if len(origins) > 1:
            return "more than one Origin header", None
        origin, p = origins[0].strip(), parse_origin(origins[0])
        if p is not None and (origin_key(origin) in self.origins or self.same(p, auth)):
            return None, origin
        shown = origin_key(origin) if p is not None else repr(origin[:100])
        return (f"Origin {shown} is not allowed (requests from web pages of other origins are refused): start the server "
                f"with --cors-origin {shown} (env MK_CORS_ORIGINS) to accept it"), None

    def authorized(self, command, path, headers):
        """the request may be answered: no API key set, /health or a CORS preflight (browsers send no credentials with
        those), or the key as `Authorization: Bearer <key>` (OpenAI's clients) or `x-api-key: <key>` (Anthropic's)"""
        if self.api_key is None or command == "OPTIONS" or path.split("?")[0] in ("/health", "/v1/health"):
            return True
        auth = (headers.get("Authorization") or "").strip()
        given = [auth[7:].strip()] if auth[:7].lower() == "bearer " else []
        given.append((headers.get("x-api-key") or "").strip())
        return any(hmac.compare_digest(g.encode(), self.api_key) for g in given if g)

    @staticmethod
    def same(p, auth):
        """the origin p is the server's own: Host's name and port (80 when Host has none). The scheme does not matter:
        nothing else can serve pages on the server's port."""
        return auth is not None and p[2] is not None and p[1] == auth[0] and p[2] == (auth[1] or 80)


def exposure_warning(host, api_key):
    """the startup warning for a listener other machines may reach with no API key; None when it is not one"""
    name = host.lower().rstrip(".").strip("[]")
    if api_key or name == "localhost":
        return None
    try:
        if ipaddress.ip_address(name).is_loopback:
            return None
    except ValueError:
        pass
    return (f"listening on {host} with no API key: anyone who can reach this port can use the model and read its "
            "replies. Set --api-key / MK_API_KEY, or listen on 127.0.0.1 only")


class BodyBudget:
    """the bytes of request bodies in memory, over all connections (MK_MAX_INFLIGHT_MB): a body counts as it arrives, not
    by the size it declares, until its request ends. It bounds the JSON parsed from them too (24 times the bytes at
    worst, measured: [[],[],...])"""

    def __init__(self, cap):
        self.cap, self.used, self.lock = cap, 0, threading.Lock()

    def charge(self, n):
        """True if n more bytes fit; they count from now"""
        with self.lock:
            if self.used + n > self.cap:
                return False
            self.used += n
            return True

    def release(self, n):
        with self.lock:
            self.used -= n


class RequestReader(io.RawIOBase):
    """a connection's reads, under the handler's rfile. MK_TIMEOUT_S limits each read, which a client sending a byte at a
    time never hits; this limits the whole arrival too: from a request's first byte until done(), at most `limit` seconds
    (0: no limit). Past it a TimeoutError, on which BaseHTTPRequestHandler closes the connection."""

    def __init__(self, sock, limit):
        self.sock, self.raw, self.limit = sock, sock.makefile("rb", 0), limit
        self.idle, self.deadline = sock.gettimeout(), None

    def readable(self):
        return True

    def readinto(self, b):
        try:
            if self.deadline is not None:
                left = self.deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("the request's deadline")
                self.sock.settimeout(left if self.idle is None else min(left, self.idle))
            n = self.raw.readinto(b)
        except TimeoutError:
            if self.deadline is not None:  # (not an idle keep-alive connection: that is routine)
                log("a request stalled or took too long to arrive (MK_TIMEOUT_S, MK_REQUEST_TIMEOUT_S): closing it")
            raise
        if n and self.limit and self.deadline is None:
            self.deadline = time.monotonic() + self.limit
        return n

    def done(self):
        """the request has arrived: the answer's writes and the next read have the plain timeout again"""
        if self.deadline is not None:
            self.deadline = None
            self.sock.settimeout(self.idle)

    def close(self):
        self.raw.close()
        super().close()


class GatedHandler(BaseHTTPRequestHandler):
    """the handlers' base: the server's Gate decides on each request before it is handled (and before a 100 Continue);
    a refused one gets a 403 and the connection closes (its body is not read). An admitted browser origin gets
    Access-Control-Allow-Origin + Vary: Origin on every response (errors and event streams too), and its CORS preflights
    (OPTIONS) are answered. A request with a body its handler does not read (any method but POST: a GET, an OPTIONS)
    is answered with Connection: close and the connection closes, so that the body is not taken for the next request
    (behind a proxy: a smuggled one)."""
    protocol_version = "HTTP/1.1"
    cors = None  # the request's admitted Origin: CORS headers on the response
    admitted = None  # the gate's verdict on the request (None: not asked yet)
    BODY_METHODS = ("POST",)  # the methods whose handlers read the request body
    unread = False  # the request has a body its handler does not read: the response closes the connection

    def log_message(self, *a):
        pass

    def setup(self):
        super().setup()
        try:  # (a stream's small events go out as they are written, not held for the client's delayed ACK: Nagle)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # (not a TCP socket)
            pass
        self.rfile.close()
        self.reader = RequestReader(self.connection, mk_limits.REQUEST_TIMEOUT_S)
        self.rfile = io.BufferedReader(self.reader)

    def handle_one_request(self):
        self.cors = self.admitted = None
        self.unread = False
        super().handle_one_request()
        self.reader.done()

    def parse_request(self):
        if not (super().parse_request() and self.admit()):
            return False
        cl = [v.strip() for v in self.headers.get_all("Content-Length") or []]  # (each one: "0" then "80" is a body)
        self.unread = self.command not in self.BODY_METHODS and bool(self.headers.get_all("Transfer-Encoding")
                                                                      or any(v != "0" for v in cl))
        if self.command not in self.BODY_METHODS:
            self.reader.done()
        return True

    def admit(self):
        """True if the gate lets the request through (asked once per request); else it is answered with a 403 (host,
        origin) or a 401 (API key)"""
        if self.admitted is None:
            gate = getattr(self.server, "gate", None)
            why, self.cors = gate.check(self.headers) if gate else (None, None)
            code, kind = 403, "permission_error"
            if why is None and gate and not gate.authorized(self.command, self.path, self.headers):
                why = "invalid or missing API key (the server was started with --api-key / MK_API_KEY)"
                code, kind = 401, "authentication_error"
            self.admitted = why is None
            if why:
                path = self.path.split("?")[0]
                log(f"{self.command} {path[:200]} -> {code}: {why}")
                err = ({"type": "error", "error": {"type": kind, "message": why}} if path.startswith("/v1/messages")
                       else {"error": {"code": code, "message": why, "type": kind}})
                data = json.dumps(err).encode()
                self.send_response(code)
                if code == 401:
                    self.send_header("WWW-Authenticate", "Bearer")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                self.wfile.write(data)
        return self.admitted

    def end_headers(self):
        if self.cors:
            self.send_header("Access-Control-Allow-Origin", self.cors)
            self.send_header("Vary", "Origin")
        if self.unread and not self.close_connection:  # (send_header sets close_connection: once)
            self.send_header("Connection", "close")
        super().end_headers()

    def do_OPTIONS(self):
        """a CORS preflight of an admitted origin (the others got the gate's 403): the methods, the headers it asked for
        and how long it may keep the answer; any other OPTIONS: the methods"""
        headers = [("Allow", CORS_METHODS)]
        if self.cors and self.headers.get("Access-Control-Request-Method"):
            headers += [("Access-Control-Allow-Methods", CORS_METHODS), ("Access-Control-Max-Age", str(CORS_MAX_AGE))]
            want = (self.headers.get("Access-Control-Request-Headers") or "").strip()
            if want and HEADER_LIST_RE.fullmatch(want):
                headers.append(("Access-Control-Allow-Headers", want))
            if (self.headers.get("Access-Control-Request-Private-Network") or "").strip().lower() == "true":
                headers.append(("Access-Control-Allow-Private-Network", "true"))  # (Chrome: a public page to this machine)
        self.send_response(200)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()
