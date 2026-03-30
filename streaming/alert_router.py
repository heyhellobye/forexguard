"""
Alert router — unified interface over Kafka and/or RabbitMQ.
Degrades silently if neither broker is running.
"""
import logging
import os

log = logging.getLogger("forexguard.router")

_SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class AlertRouter:
    def __init__(self, kafka_bootstrap=None, rabbitmq_host=None,
                 rabbitmq_port=None, rabbitmq_user=None, rabbitmq_pass=None,
                 min_severity=None, enable_kafka=True, enable_rabbitmq=True):
        self._kafka_bootstrap = (kafka_bootstrap or
                                 os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
        self._rmq_host  = (rabbitmq_host or os.getenv("RABBITMQ_HOST", "localhost"))
        self._rmq_port  = int(rabbitmq_port or os.getenv("RABBITMQ_PORT", "5672"))
        self._rmq_user  = (rabbitmq_user or os.getenv("RABBITMQ_USER", "guest"))
        self._rmq_pass  = (rabbitmq_pass or os.getenv("RABBITMQ_PASS", "guest"))
        self._min_rank  = _SEVERITY_RANK.get(
            (min_severity or os.getenv("ALERT_MIN_SEVERITY", "MEDIUM")).upper(), 1)

        self._kafka_pub  = None
        self._rmq_pub    = None
        self._published  = 0
        self._suppressed = 0

        if enable_kafka:
            self._init_kafka()
        if enable_rabbitmq:
            self._init_rabbitmq()

        self._log_status()

    def _init_kafka(self):
        try:
            from streaming.kafka_publisher import (KafkaAlertPublisher,
                                                    ensure_topics, _probe_broker)
            if _probe_broker(self._kafka_bootstrap):
                ensure_topics(self._kafka_bootstrap)
                self._kafka_pub = KafkaAlertPublisher(self._kafka_bootstrap)
            else:
                log.warning("[router] Kafka broker not reachable — skipping")
        except Exception as e:
            log.warning(f"[router] Kafka init failed: {e}")

    def _init_rabbitmq(self):
        try:
            from streaming.rabbitmq_publisher import RabbitMQAlertPublisher
            self._rmq_pub = RabbitMQAlertPublisher(
                host=self._rmq_host, port=self._rmq_port,
                username=self._rmq_user, password=self._rmq_pass)
        except Exception as e:
            log.warning(f"[router] RabbitMQ init failed: {e}")

    def _log_status(self):
        kafka_ok = self._kafka_pub and self._kafka_pub.available
        rmq_ok   = self._rmq_pub   and self._rmq_pub.available
        log.info(f"[router] Kafka={'OK' if kafka_ok else 'unavailable'}  "
                 f"RabbitMQ={'OK' if rmq_ok else 'unavailable'}")

    def publish(self, alert: dict) -> dict:
        severity = alert.get("severity", "LOW").upper()
        rank     = _SEVERITY_RANK.get(severity, 0)

        if rank < self._min_rank:
            self._suppressed += 1
            return {"kafka": False, "rabbitmq": False, "suppressed": True}

        kafka_ok = False
        rmq_ok   = False

        if self._kafka_pub and self._kafka_pub.available:
            kafka_ok = self._kafka_pub.publish(alert)

        if self._rmq_pub and self._rmq_pub.available:
            rmq_ok = self._rmq_pub.publish(alert)

        if kafka_ok or rmq_ok:
            self._published += 1

        return {"kafka": kafka_ok, "rabbitmq": rmq_ok, "suppressed": False}

    def stats(self) -> dict:
        return {
            "published":       self._published,
            "suppressed":      self._suppressed,
            "kafka_available": bool(self._kafka_pub and self._kafka_pub.available),
            "rmq_available":   bool(self._rmq_pub   and self._rmq_pub.available),
        }

    def close(self):
        if self._kafka_pub:
            self._kafka_pub.close()
        if self._rmq_pub:
            self._rmq_pub.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()