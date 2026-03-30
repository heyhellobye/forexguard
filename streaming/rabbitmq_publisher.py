"""
RabbitMQ alert publisher.

Exchange topology
─────────────────
  Exchange : forexguard  (topic exchange)
  Routing keys:
    forexguard.alerts.login
    forexguard.alerts.trading
    forexguard.alerts.deposit
    forexguard.alerts.withdrawal
    forexguard.alerts.critical   (severity=CRITICAL, any type)
    forexguard.alerts.all        (every alert)

Compliance team can bind queues with wildcards:
  forexguard.alerts.#          → all alerts
  forexguard.alerts.critical   → CRITICAL only
  forexguard.alerts.login      → login anomalies only

Requires:  pip install pika

Degrades gracefully if broker is unreachable.
"""
import json
import logging
import threading
from datetime import datetime

log = logging.getLogger("forexguard.rabbitmq")

EXCHANGE     = "forexguard"
EXCHANGE_TYPE= "topic"


def _routing_key(alert: dict) -> str:
    severity = alert.get("severity", "LOW").upper()
    if severity == "CRITICAL":
        return "forexguard.alerts.critical"
    flags = " ".join(alert.get("flags", [])).lower()
    if any(k in flags for k in ["login","ip","travel","brute","device"]):
        return "forexguard.alerts.login"
    if any(k in flags for k in ["trade","volume","instrument","pnl","news"]):
        return "forexguard.alerts.trading"
    if any(k in flags for k in ["deposit","structuring","bonus","micro"]):
        return "forexguard.alerts.deposit"
    if any(k in flags for k in ["withdrawal","dormancy","cluster","kyc"]):
        return "forexguard.alerts.withdrawal"
    return "forexguard.alerts.all"


class RabbitMQAlertPublisher:
    def __init__(self, host: str = "localhost", port: int = 5672,
                 username: str = "guest", password: str = "guest",
                 vhost: str = "/"):
        self.host       = host
        self.port       = port
        self.username   = username
        self.password   = password
        self.vhost      = vhost
        self._connection = None
        self._channel    = None
        self._available  = False
        self._lock       = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            import pika
            credentials = pika.PlainCredentials(self.username, self.password)
            params      = pika.ConnectionParameters(
                host        = self.host,
                port        = self.port,
                virtual_host= self.vhost,
                credentials = credentials,
                socket_timeout      = 3,
                connection_attempts = 2,
                retry_delay         = 1,
            )
            self._connection = pika.BlockingConnection(params)
            self._channel    = self._connection.channel()

            # Declare durable topic exchange
            self._channel.exchange_declare(
                exchange      = EXCHANGE,
                exchange_type = EXCHANGE_TYPE,
                durable       = True,
            )

            # Declare and bind convenience queues
            self._setup_queues()

            self._available = True
            log.info(f"[rabbitmq] connected to {self.host}:{self.port}")

        except ImportError:
            log.warning("[rabbitmq] pika not installed. Run: pip install pika")
        except Exception as e:
            log.warning(f"[rabbitmq] broker unavailable ({e}) — "
                        "alerts will not be published to RabbitMQ")

    def _setup_queues(self):
        """Declare standard compliance queues and bind them."""
        queues = [
            ("forexguard.q.all",        "forexguard.alerts.#"),
            ("forexguard.q.critical",   "forexguard.alerts.critical"),
            ("forexguard.q.login",      "forexguard.alerts.login"),
            ("forexguard.q.trading",    "forexguard.alerts.trading"),
            ("forexguard.q.deposit",    "forexguard.alerts.deposit"),
            ("forexguard.q.withdrawal", "forexguard.alerts.withdrawal"),
        ]
        for queue_name, routing_key in queues:
            self._channel.queue_declare(queue=queue_name, durable=True)
            self._channel.queue_bind(
                exchange    = EXCHANGE,
                queue       = queue_name,
                routing_key = routing_key,
            )
            log.debug(f"[rabbitmq] queue bound: {queue_name} <- {routing_key}")

    def publish(self, alert: dict) -> bool:
        if not self._available or self._channel is None:
            return False
        try:
            import pika
            with self._lock:
                routing_key = _routing_key(alert)
                payload     = json.dumps({
                    **alert,
                    "published_at": datetime.utcnow().isoformat() + "Z",
                    "source":       "forexguard-engine",
                    "broker":       "rabbitmq",
                }).encode("utf-8")

                self._channel.basic_publish(
                    exchange    = EXCHANGE,
                    routing_key = routing_key,
                    body        = payload,
                    properties  = pika.BasicProperties(
                        delivery_mode = 2,         # persistent message
                        content_type  = "application/json",
                        headers       = {
                            "user_id":  alert.get("user_id", ""),
                            "severity": alert.get("severity", ""),
                            "score":    str(alert.get("ensemble_score", 0)),
                        },
                    ),
                )
                log.debug(f"[rabbitmq] published → {routing_key} | "
                          f"user={alert.get('user_id')} "
                          f"severity={alert.get('severity')}")
                return True
        except Exception as e:
            log.warning(f"[rabbitmq] publish error: {e}")
            self._available = False   # stop retrying broken connection
            return False

    def close(self):
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available
