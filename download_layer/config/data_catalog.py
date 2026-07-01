from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DownloadAsset:
    key: str
    name: str
    status: str
    owner_layer: str
    source: str
    target_layer: str
    target_table: str
    artifact_area: str
    frequencies: tuple[str, ...] = ()
    adjust_modes: tuple[str, ...] = ()
    frontend_enabled: bool = False
    notes: str = ""


DOWNLOAD_CATALOG: tuple[DownloadAsset, ...] = (
    DownloadAsset(
        key="ashare_kline",
        name="A-share K-line bars",
        status="implemented",
        owner_layer="download_layer",
        source="BaoStock",
        target_layer="data_layer",
        target_table="kline_bars",
        artifact_area="runtime_layer/data",
        frequencies=("daily", "weekly", "60min", "30min", "15min", "5min"),
        adjust_modes=("post", "pre", "none"),
        frontend_enabled=True,
        notes=(
            "Core cross-frequency OHLCV bars. 5min is the preferred future feature base; pctChg is calculated locally by data_layer; "
            "turnover and trading status are stored in extension tables."
        ),
    ),
    DownloadAsset(
        key="trade_calendar",
        name="Trading calendar",
        status="implemented",
        owner_layer="download_layer",
        source="BaoStock",
        target_layer="data_layer",
        target_table="trade_calendar",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Required prerequisite for coverage checks and feature progress calculation.",
    ),
    DownloadAsset(
        key="stock_liquidity_daily",
        name="Daily stock liquidity",
        status="implemented",
        owner_layer="download_layer",
        source="BaoStock",
        target_layer="data_layer",
        target_table="stock_liquidity_daily",
        artifact_area="runtime_layer/data",
        frequencies=("daily",),
        frontend_enabled=False,
        notes="Daily turnover facts. Used later by feature_layer for model input alignment.",
    ),
    DownloadAsset(
        key="stock_status_daily",
        name="Historical ST status",
        status="implemented",
        owner_layer="download_layer",
        source="BaoStock",
        target_layer="data_layer",
        target_table="stock_status_daily",
        artifact_area="runtime_layer/data",
        frequencies=("daily",),
        frontend_enabled=False,
        notes="Required dated market-rule fact: ST days use a 5% limit and block new buys; never emitted as a model feature.",
    ),
    DownloadAsset(
        key="symbol_master",
        name="A-share symbol master",
        status="planned",
        owner_layer="download_layer",
        source="BaoStock / future providers",
        target_layer="data_layer",
        target_table="symbol_master",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Symbol metadata: symbol, name, exchange, listing date, delisting date, status.",
    ),
    DownloadAsset(
        key="index_kline",
        name="Market index K-line bars",
        status="planned",
        owner_layer="download_layer",
        source="BaoStock / future providers",
        target_layer="data_layer",
        target_table="index_bars",
        artifact_area="runtime_layer/data",
        frequencies=("daily", "weekly", "60min", "30min"),
        adjust_modes=("none",),
        frontend_enabled=False,
        notes="Market context indices such as SSE Composite, CSI 300, CSI 500, ChiNext.",
    ),
    DownloadAsset(
        key="tradability",
        name="Tradability state",
        status="planned",
        owner_layer="download_layer",
        source="BaoStock / future providers",
        target_layer="data_layer",
        target_table="stock_tradability",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Suspension, ST status, limit-up/limit-down state, and other tradability constraints.",
    ),
    DownloadAsset(
        key="corporate_actions",
        name="Corporate actions and adjustment factors",
        status="planned",
        owner_layer="download_layer",
        source="BaoStock / future providers",
        target_layer="data_layer",
        target_table="corporate_actions",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Dividends, splits, adjustment factors, and ex-right information.",
    ),
    DownloadAsset(
        key="market_cap",
        name="Shares and market capitalization",
        status="planned",
        owner_layer="download_layer",
        source="BaoStock / future providers",
        target_layer="data_layer",
        target_table="market_cap_daily",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Total shares, float shares, market cap, float market cap. Needed for turnover and market-cap normalization.",
    ),
    DownloadAsset(
        key="flow_data",
        name="Order flow / money flow",
        status="planned",
        owner_layer="download_layer",
        source="future providers",
        target_layer="data_layer",
        target_table="flow_bars",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Future extension for money flow, order imbalance, and market-cap/trading-value normalized flow features.",
    ),
    DownloadAsset(
        key="fundamentals",
        name="Fundamental data",
        status="planned",
        owner_layer="download_layer",
        source="future providers",
        target_layer="data_layer",
        target_table="fundamentals_daily",
        artifact_area="runtime_layer/data",
        frontend_enabled=False,
        notes="Revenue, profit, PE, PB, ROE, debt ratio, and other slow-moving features.",
    ),
)


def list_download_assets() -> list[dict[str, Any]]:
    return [asdict(asset) for asset in DOWNLOAD_CATALOG]


def get_download_asset(key: str) -> dict[str, Any]:
    normalized = str(key).strip().lower()
    for asset in DOWNLOAD_CATALOG:
        if asset.key == normalized:
            return asdict(asset)
    raise KeyError(f"Unknown download asset: {key}")


def catalog_payload() -> dict[str, Any]:
    assets = list_download_assets()
    implemented = [asset for asset in assets if asset["status"] == "implemented"]
    planned = [asset for asset in assets if asset["status"] != "implemented"]

    return {
        "version": "download_catalog_v1",
        "implemented_count": len(implemented),
        "planned_count": len(planned),
        "frontend_enabled": [
            asset for asset in assets
            if bool(asset.get("frontend_enabled"))
        ],
        "implemented": implemented,
        "planned": planned,
        "assets": assets,
    }
