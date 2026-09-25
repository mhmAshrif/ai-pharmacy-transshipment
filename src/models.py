from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Text,
    JSON,
    func,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # map ORM attribute names to the existing DB column names used in your DDL
    district = Column("district", String(100), nullable=False, index=True)
    medicine_name = Column("medicine", String(100), nullable=False, index=True)
    category = Column("category", String(100), nullable=True)
    # DB uses `stock_level` in the existing schema
    stock_quantity = Column("stock_level", Integer, default=0, nullable=False)
    expiry_days_remaining = Column(Integer, nullable=True, index=True)
    # DB stores unit_price as numeric(10,2) — use Numeric for exactness
    from sqlalchemy import Numeric
    unit_price = Column(Numeric(10, 2), default=0.0, nullable=False)


class TransferManifest(Base):
    __tablename__ = "transfer_manifests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Map to existing DB columns which use *_id foreign keys
    source_district_id = Column("source_district_id", Integer, nullable=True, index=True)
    dest_district_id = Column("dest_district_id", Integer, nullable=True, index=True)
    # Medicine FK
    medicine_id = Column("medicine_id", Integer, nullable=True, index=True)
    # Quantity and costs
    quantity_to_move = Column("quantity_to_move", Integer, default=0, nullable=False)
    transport_cost = Column("transport_cost", Float, default=0.0, nullable=True)
    expiring_asset_value = Column("expiring_asset_value", Float, default=0.0, nullable=True)
    net_savings = Column("net_savings", Float, default=0.0, nullable=True)
    status = Column(String(32), default="PENDING_DISPATCH", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(128), nullable=False, index=True)
    entity_id = Column(Integer, nullable=True)
    user_id = Column(String(128), nullable=True)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ForecastMetric(Base):
    __tablename__ = "forecast_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    district = Column(String(128), nullable=False, index=True)
    medicine_name = Column(String(256), nullable=False, index=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    best_changepoint_prior = Column(Float, nullable=True)
    best_seasonality_prior = Column(Float, nullable=True)
    evaluated_at = Column(DateTime(timezone=True), server_default=func.now())


class District(Base):
    __tablename__ = "districts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True)


class Medicine(Base):
    __tablename__ = "medicines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False, index=True)
