"""
Kafka / Redpanda alert publisher.
Degrades silently if broker is not running.
"""
import json
import logging
import os
import sys
from datetime import datetime

log = logging.getLogger("forexguard.kafka")

TOPICS = {
    "login":      "forexguard.alerts.login",
    "trading":    "forexguard.alerts.trading",
    "deposit":    "forexguard.alerts.deposit",
    "withdrawal": "forexguard.alerts.withdrawal",
    "default":    "forexguard.alerts",
}


def _get_topic(alert: dict) -> str:
    flags = " ".join(alert.get("flags", [])).lower()
    if any(k in flags for k in ["login","ip","travel","brute","device"]):
        return TOPICS["login"]
    if any(k in flags for k in ["trade","volume","instrument","pnl","news"]):
        return TOPICS["trading"]
    if any(k in flags for k in ["deposit","structuring","bonus","micro"]):
        return TOPICS["deposit"]
    if any(k in flags for k in ["withdrawal","dormancy","cluster","kyc"]):
        return TOPICS["withdrawal"]
    return TOPICS["default"]


def _probe_broker(bootstrap_servers: str, timeout_ms: int = 3000) -> bool:
    """Quick TCP probe before creating a Producer — avoids background retry spam."""
    try:
        host, port = bootstrap_servers.split(":")[0], int(bootstrap_servers.split(":")[1])
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout_ms / 1000)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception:
        return False


class KafkaAlertPublisher:
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer  = None
        self._available = False
        self._connect()

    def _connect(self):
        # TCP probe first — if broker not reachable, skip entirely
        if not _probe_broker(self.bootstrap_servers):
            log.warning("[kafka] broker not reachable — running without Kafka")
            return
        try:
            from confluent_kafka import Producer
            # Silence confluent-kafka's internal C-level stderr completely
            devnull = open(os.devnull, 'w')
            old_stderr = sys.stderr
            sys.stderr = devnull
            conf = {
                "bootstrap.servers":      self.bootstrap_servers,
                "client.id":              "forexguard-alert-publisher",
                "acks":                   "1",
                "retries":                0,
                "socket.timeout.ms":      3000,
                "message.timeout.ms":     5000,
                "log_level":              0,
            }
            self._producer  = Producer(conf)
            sys.stderr = old_stderr
            self._available = True
            log.info(f"[kafka] connected to {self.bootstrap_servers}")
        except ImportError:
            log.warning("[kafka] confluent-kafka not installed")
        except Exception as e:
            log.warning(f"[kafka] init failed: {e}")

    def _delivery_report(self, err, msg):
        if err:
            log.debug(f"[kafka] delivery failed: {err}")

    def publish(self, alert: dict) -> bool:
        if not self._available or self._producer is None:
            return False
        try:
            topic   = _get_topic(alert)
            payload = json.dumps({
                **alert,
                "published_at": datetime.utcnow().isoformat() + "Z",
                "source":       "forexguard-engine",
            }).encode("utf-8")
            self._producer.produce(
                topic    = topic,
                key      = alert.get("user_id", "unknown").encode("utf-8"),
                value    = payload,
                callback = self._delivery_report,
            )
            self._producer.poll(0)
            return True
        except Exception as e:
            log.warning(f"[kafka] publish error: {e}")
            self._available = False
            return False

    def flush(self, timeout: float = 2.0):
        if self._available and self._producer:
            try:
                self._producer.flush(timeout)
            except Exception:
                pass

    def close(self):
        self.flush()

    @property
    def available(self) -> bool:
        return self._available


def ensure_topics(bootstrap_servers: str = "localhost:9092"):
    """Only attempts topic creation if broker is actually reachable."""
    if not _probe_broker(bootstrap_servers):
        log.debug("[kafka] skipping topic creation — broker not reachable")
        return
    try:
        from confluent_kafka.admin import AdminClient, NewTopic
        admin    = AdminClient({"bootstrap.servers": bootstrap_servers,
                                "socket.timeout.ms": 3000})
        futures  = admin.create_topics([
            NewTopic(t, num_partitions=3, replication_factor=1)
            for t in TOPICS.values()
        ])
        for topic, future in futures.items():
            try:
                future.result()
                log.info(f"[kafka] topic created: {topic}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    log.debug(f"[kafka] topic note {topic}: {e}")
    except Exception as e:
        log.debug(f"[kafka] ensure_topics: {e}")