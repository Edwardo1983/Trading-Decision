from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import Candle, IndicatorResult, MarketRegime
from core.timeframes import to_seconds

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Structure for parsing AI response (manual input after running prompt)."""
    timestamp: datetime
    market_sentiment: str  # bullish, bearish, neutral
    confidence: float  # 0.0 - 1.0
    key_observations: List[str]
    recommended_action: str  # LONG, SHORT, WAIT, CLOSE_POSITIONS
    risk_level: str  # low, moderate, high, extreme
    reasoning: str


@dataclass
class PromptTimeframeContext:
    timeframe: str
    candles: List[Candle]
    indicators: List[IndicatorResult]
    sentiment: Dict[str, Any] = None
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    day_classification: str = "unknown"
    patterns: List[Dict[str, Any]] = None
    note: str = ""

    @classmethod
    def from_value(cls, timeframe: str, value: Any) -> "PromptTimeframeContext":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"Unsupported timeframe context type for {timeframe}: {type(value)!r}")
        return cls(
            timeframe=timeframe,
            candles=list(value.get("candles", [])),
            indicators=list(value.get("indicators", [])),
            sentiment=dict(value.get("sentiment", {}) or {}),
            market_regime=value.get("market_regime", MarketRegime.UNKNOWN),
            day_classification=str(value.get("day_classification", "unknown")),
            patterns=list(value.get("patterns", [])),
            note=str(value.get("note", "")),
        )


@dataclass
class PromptBundleArtifact:
    symbol: str
    generated_at: datetime
    primary_timeframe: str
    timeframe_contexts: Dict[str, PromptTimeframeContext]
    prompts: Dict[str, str]
    latest_prompt_path: Path
    latest_bundle_txt_path: Path
    latest_bundle_json_path: Path
    archive_bundle_path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "primary_timeframe": self.primary_timeframe,
            "timeframes": {
                tf: {
                    "timeframe": ctx.timeframe,
                    "candles": len(ctx.candles),
                    "indicators": len(ctx.indicators),
                    "market_regime": ctx.market_regime.value if hasattr(ctx.market_regime, "value") else str(ctx.market_regime),
                    "day_classification": ctx.day_classification,
                    "note": ctx.note,
                }
                for tf, ctx in self.timeframe_contexts.items()
            },
            "prompts": self.prompts,
            "paths": {
                "latest_prompt": str(self.latest_prompt_path),
                "latest_bundle_txt": str(self.latest_bundle_txt_path),
                "latest_bundle_json": str(self.latest_bundle_json_path),
                "archive_bundle": str(self.archive_bundle_path),
            },
        }


class PromptGenerator:
    """
    Generates structured prompts for manual execution on Claude Code CLI or Codex.

    Usage:
        1. Call generate_prompt() to get the formatted prompt
        2. Copy the prompt to Claude Code CLI or Codex
        3. Paste the AI response back using parse_response()

    The prompt is saved to a file for easy access.
    """

    # Main analysis prompt template - legacy single-timeframe mode.
    ANALYSIS_PROMPT_TEMPLATE = '''You are a professional cryptocurrency/trading analyst with expertise in technical analysis.
Analyze the following market data and provide an objective evaluation.

Target AI: {target_name}

## CURRENT MARKET DATA

### Symbol: {symbol}
### Timestamp: {timestamp}

### Price and Candles (last {lookback} bars)
{candle_data}

### Technical Indicators Summary
{indicators}

### Market Sentiment Data
{sentiment}

### Current Market Regime
{market_regime}

### Day Classification
{day_classification}

### Pattern Detection
{patterns}

---

## ANALYSIS REQUIREMENTS

Based on ALL the data above, provide your analysis in the following JSON format:

```json
{{
    "market_sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "key_observations": [
        "observation 1",
        "observation 2",
        "observation 3"
    ],
    "recommended_action": "LONG" | "SHORT" | "WAIT" | "CLOSE_POSITIONS",
    "risk_level": "low" | "moderate" | "high" | "extreme",
    "reasoning": "Detailed explanation of your analysis and reasoning"
}}
```

## GUIDELINES

1. Consider the confluence of multiple indicators
2. Weight trend indicators heavily for direction
3. Use momentum for timing and strength
4. Check volume for confirmation
5. Respect the market regime (trending vs ranging)
6. Be conservative - when in doubt, recommend WAIT
7. Provide SPECIFIC observations, not generic ones

Respond ONLY with the JSON, no additional text.'''

    MULTI_TIMEFRAME_ANALYSIS_PROMPT_TEMPLATE = '''You are a professional cryptocurrency/trading analyst with expertise in technical analysis.
Analyze the following market data and provide an objective evaluation.

Target AI: {target_name}

## CURRENT MARKET DATA

### Symbol: {symbol}
### Timestamp: {timestamp}
### Primary Timeframe: {primary_timeframe}
### Included Timeframes: {included_timeframes}

## MULTI-TIMEFRAME CONTEXT

### Cross-Timeframe Summary
{bundle_overview}

### Timeframe Details
{timeframe_bundle}

### Shared Sentiment Data
{sentiment}

### Current Market Regime
{market_regime}

### Day Classification
{day_classification}

### Pattern Detection
{patterns}

---

## ANALYSIS REQUIREMENTS

Based on ALL the data above, provide your analysis in the following JSON format:

```json
{{
    "market_sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "key_observations": [
        "observation 1",
        "observation 2",
        "observation 3"
    ],
    "recommended_action": "LONG" | "SHORT" | "WAIT" | "CLOSE_POSITIONS",
    "risk_level": "low" | "moderate" | "high" | "extreme",
    "reasoning": "Detailed explanation of your analysis and reasoning"
}}
```

## GUIDELINES

1. Consider the confluence of multiple indicators across all timeframes
2. Weight higher timeframes for bias and lower timeframes for execution timing
3. Use momentum for timing and strength
4. Check volume for confirmation
5. Respect the market regime (trending vs ranging)
6. Be conservative - when in doubt, recommend WAIT
7. Provide SPECIFIC observations, not generic ones

Respond ONLY with the JSON, no additional text.'''

    def __init__(
        self,
        output_dir: str = "prompts",
        lookback: int = 21,
        targets: Optional[List[str]] = None,
    ):
        """
        Initialize prompt generator.

        Args:
            output_dir: Directory to save generated prompts
            lookback: Number of candles to include in prompt
        """
        self.output_dir = Path(output_dir)
        self.lookback = lookback
        self.targets = [str(t).lower() for t in (targets or ["claude", "codex"]) if str(t).strip()]
        if not self.targets:
            self.targets = ["claude", "codex"]
        self._last_prompt: Optional[str] = None
        self._last_prompt_time: Optional[datetime] = None
        self._last_result: Optional[AnalysisResult] = None
        self._last_bundle: Optional[Dict[str, Any]] = None

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timeframe_sort_key(timeframe: str) -> tuple[int, str]:
        try:
            return to_seconds(timeframe), timeframe
        except Exception:
            return 10**9, timeframe

    @staticmethod
    def _normalize_market_regime(value: Any) -> MarketRegime:
        if isinstance(value, MarketRegime):
            return value
        try:
            return MarketRegime(str(value))
        except Exception:
            return MarketRegime.UNKNOWN

    @staticmethod
    def _normalize_patterns(patterns: Any) -> List[Dict[str, Any]]:
        if not patterns:
            return []
        if isinstance(patterns, list):
            return [item for item in patterns if isinstance(item, dict)]
        return []

    @staticmethod
    def _count_states(indicators: List[IndicatorResult]) -> Dict[str, int]:
        counts = {"BUY": 0, "SELL": 0, "NEUTRAL": 0, "NO_TRADE": 0, "WAIT": 0}
        for indicator in indicators:
            key = getattr(indicator.state, "value", str(indicator.state))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _format_candle_data(self, candles: List[Candle]) -> str:
        """Format candle data for the prompt."""
        last_n = candles[-self.lookback:] if len(candles) >= self.lookback else candles
        if not last_n:
            return "  No candle data available"

        lines = []
        lines.append("  | # | Time | Open | High | Low | Close | Volume | Change |")
        lines.append("  |---|------|------|------|-----|-------|--------|--------|")

        prev_close = None
        for i, c in enumerate(last_n):
            ts = c.timestamp.strftime("%H:%M") if hasattr(c.timestamp, "strftime") else str(c.timestamp)[-5:]

            # Calculate change from previous
            if prev_close:
                change_pct = ((c.close - prev_close) / prev_close) * 100
                change_str = f"{change_pct:+.2f}%"
            else:
                change_str = "-"
            prev_close = c.close

            lines.append(
                f"  | {i+1:2d} | {ts} | {c.open:.2f} | {c.high:.2f} | {c.low:.2f} | {c.close:.2f} | {c.volume:.0f} | {change_str} |"
            )

        # Add summary
        if last_n:
            first_close = last_n[0].close
            last_close = last_n[-1].close
            total_change = ((last_close - first_close) / first_close) * 100
            high = max(c.high for c in last_n)
            low = min(c.low for c in last_n)
            avg_vol = sum(c.volume for c in last_n) / len(last_n)

            lines.append("")
            lines.append(f"  Summary: Price {first_close:.2f} -> {last_close:.2f} ({total_change:+.2f}%)")
            lines.append(f"  Range: {low:.2f} - {high:.2f} | Avg Volume: {avg_vol:.0f}")

        return "\n".join(lines)

    def _format_timeframe_context(self, timeframe: str, ctx: PromptTimeframeContext) -> str:
        sections = [
            f"  ### {timeframe}",
            f"  Regime: {ctx.market_regime.value if hasattr(ctx.market_regime, 'value') else str(ctx.market_regime)}",
            f"  Day Classification: {ctx.day_classification}",
        ]
        if ctx.note:
            sections.append(f"  Note: {ctx.note}")
        sections.append("")
        sections.append(self._format_candle_data(ctx.candles))
        sections.append("")
        sections.append(self._format_indicators(ctx.indicators))
        sections.append("")
        sections.append(self._format_sentiment(ctx.sentiment or {}))
        sections.append("")
        sections.append(self._format_patterns(self._normalize_patterns(ctx.patterns)))
        return "\n".join(sections)

    def _format_bundle_overview(self, timeframe_contexts: Dict[str, PromptTimeframeContext]) -> str:
        if not timeframe_contexts:
            return "  No timeframe contexts available"

        lines: List[str] = []
        for timeframe in sorted(timeframe_contexts.keys(), key=self._timeframe_sort_key):
            ctx = timeframe_contexts[timeframe]
            candle_count = len(ctx.candles)
            last_close = ctx.candles[-1].close if ctx.candles else None
            first_close = ctx.candles[0].close if ctx.candles else None
            change = None
            if candle_count >= 2 and first_close:
                change = ((last_close - first_close) / first_close) * 100 if last_close is not None else None
            states = self._count_states(ctx.indicators)
            line = f"  - {timeframe}: candles={candle_count}"
            if last_close is not None:
                line += f", close={last_close:.2f}"
            if change is not None:
                line += f", change={change:+.2f}%"
            line += (
                f", regime={ctx.market_regime.value if hasattr(ctx.market_regime, 'value') else str(ctx.market_regime)}"
                f", buy={states.get('BUY', 0)}, sell={states.get('SELL', 0)}, neutral={states.get('NEUTRAL', 0)}"
            )
            lines.append(line)
        return "\n".join(lines)

    def _normalize_timeframe_contexts(
        self,
        timeframe_contexts: Optional[Dict[str, Any]],
    ) -> Dict[str, PromptTimeframeContext]:
        normalized: Dict[str, PromptTimeframeContext] = {}
        if not timeframe_contexts:
            return normalized
        for timeframe, value in timeframe_contexts.items():
            normalized[str(timeframe)] = PromptTimeframeContext.from_value(str(timeframe), value)
        return dict(sorted(normalized.items(), key=lambda item: self._timeframe_sort_key(item[0])))

    def _build_multi_timeframe_prompt(
        self,
        target_name: str,
        symbol: str,
        timeframe_contexts: Dict[str, PromptTimeframeContext],
        primary_timeframe: str,
        sentiment: Dict[str, Any] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        day_classification: str = "unknown",
        patterns: List[Dict[str, Any]] = None,
    ) -> str:
        bundle_overview = self._format_bundle_overview(timeframe_contexts)
        timeframe_bundle = "\n\n".join(
            self._format_timeframe_context(timeframe, ctx) for timeframe, ctx in timeframe_contexts.items()
        )
        included_timeframes = ", ".join(timeframe_contexts.keys()) if timeframe_contexts else "none"
        return self.MULTI_TIMEFRAME_ANALYSIS_PROMPT_TEMPLATE.format(
            target_name=target_name.upper(),
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            primary_timeframe=primary_timeframe,
            included_timeframes=included_timeframes,
            bundle_overview=bundle_overview,
            timeframe_bundle=timeframe_bundle,
            sentiment=self._format_sentiment(sentiment or {}),
            market_regime=market_regime.value if hasattr(market_regime, "value") else str(market_regime),
            day_classification=day_classification,
            patterns=self._format_patterns(patterns or []),
        )

    def _format_indicators(self, indicators: List[IndicatorResult]) -> str:
        """Format indicator data for the prompt."""
        if not indicators:
            return "  No indicator data available"

        # Group by category
        by_category: Dict[str, List[IndicatorResult]] = {}
        for ind in indicators:
            cat = ind.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ind)

        lines = []
        for category, inds in sorted(by_category.items()):
            lines.append(f"\n  ### {category.upper()}")
            for ind in inds:
                state_icon = {
                    "BUY": "[BUY]",
                    "SELL": "[SELL]",
                    "NEUTRAL": "[---]",
                    "NO_TRADE": "[X]",
                }.get(ind.state.value, "[?]")

                # Format value
                if isinstance(ind.value, dict):
                    # Show key metrics
                    val_items = []
                    for k, v in list(ind.value.items())[:4]:
                        if isinstance(v, float):
                            val_items.append(f"{k}={v:.2f}")
                        else:
                            val_items.append(f"{k}={v}")
                    val_str = ", ".join(val_items)
                else:
                    val_str = str(ind.value)

                lines.append(f"  - {ind.name}: {state_icon} {val_str} (conf: {ind.confidence:.0f}%)")
                if ind.reason:
                    lines.append(f"    Reason: {ind.reason}")

        return "\n".join(lines)

    def _format_sentiment(self, sentiment: Dict[str, Any]) -> str:
        """Format sentiment data for the prompt."""
        if not sentiment:
            return "  No sentiment data available"

        lines = []
        for key, value in sentiment.items():
            if value is not None:
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.2f}")
                else:
                    lines.append(f"  - {key}: {value}")

        return "\n".join(lines) if lines else "  No sentiment data available"

    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format detected patterns for the prompt."""
        if not patterns:
            return "  No patterns detected"

        lines = []
        for p in patterns:
            name = p.get("name", "Unknown")
            ptype = p.get("type", "neutral")
            conf = p.get("confidence", 0)
            icon = {"bullish": "[+]", "bearish": "[-]", "neutral": "[=]"}.get(ptype, "[?]")
            lines.append(f"  - {icon} {name} ({ptype}, confidence: {conf:.0%})")

        return "\n".join(lines)

    def _build_prompt(
        self,
        target_name: str,
        symbol: str,
        candles: List[Candle],
        indicators: List[IndicatorResult],
        sentiment: Dict[str, Any] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        day_classification: str = "unknown",
        patterns: List[Dict[str, Any]] = None,
        save_to_file: bool = True,
    ) -> str:
        """
        Generate a structured prompt for Claude Code or Codex.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            candles: List of candles
            indicators: List of indicator results
            sentiment: Sentiment data dictionary
            market_regime: Current market regime
            day_classification: Day type (AGGRESSIVE/MODERATE/CHILL)
            patterns: Detected patterns from PatternDetector
            save_to_file: Whether to save prompt to file

        Returns:
            Formatted prompt string ready to be copied to AI
        """
        return self.ANALYSIS_PROMPT_TEMPLATE.format(
            target_name=target_name.upper(),
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            lookback=self.lookback,
            candle_data=self._format_candle_data(candles),
            indicators=self._format_indicators(indicators),
            sentiment=self._format_sentiment(sentiment or {}),
            market_regime=market_regime.value if hasattr(market_regime, 'value') else str(market_regime),
            day_classification=day_classification,
            patterns=self._format_patterns(patterns or []),
        )

    def generate_prompt(
        self,
        symbol: str,
        candles: List[Candle],
        indicators: List[IndicatorResult],
        sentiment: Dict[str, Any] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        day_classification: str = "unknown",
        patterns: List[Dict[str, Any]] = None,
        save_to_file: bool = True,
        target_name: str = "claude",
    ) -> str:
        now = datetime.now(timezone.utc)
        target = str(target_name or "claude").lower()
        prompt = self._build_prompt(
            target_name=target,
            symbol=symbol,
            candles=candles,
            indicators=indicators,
            sentiment=sentiment,
            market_regime=market_regime,
            day_classification=day_classification,
            patterns=patterns,
        )

        self._last_prompt = prompt
        self._last_prompt_time = now

        if save_to_file:
            self._save_prompt(prompt, symbol, now, target)

        return prompt

    def generate_prompt_bundle(
        self,
        symbol: str,
        candles: List[Candle],
        indicators: List[IndicatorResult],
        sentiment: Dict[str, Any] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        day_classification: str = "unknown",
        patterns: List[Dict[str, Any]] = None,
        save_to_file: bool = True,
        timeframe_contexts: Optional[Dict[str, Any]] = None,
        primary_timeframe: Optional[str] = None,
    ) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        prompts: Dict[str, str] = {}
        normalized_contexts = self._normalize_timeframe_contexts(timeframe_contexts)
        if normalized_contexts:
            primary_tf = primary_timeframe if primary_timeframe in normalized_contexts else next(iter(normalized_contexts))
            artifact = self._build_prompt_bundle_artifact(
                symbol=symbol,
                timeframe_contexts=normalized_contexts,
                primary_timeframe=primary_tf,
                sentiment=sentiment,
                market_regime=market_regime,
                day_classification=day_classification,
                patterns=patterns,
                save_to_file=save_to_file,
                timestamp=now,
            )
            prompts.update(artifact.prompts)
            self._last_prompt = artifact.prompts[self.targets[0]]
            self._last_prompt_time = now
            self._last_bundle = artifact.to_dict()
            return prompts

        for target in self.targets:
            prompt = self._build_prompt(
                target_name=target,
                symbol=symbol,
                candles=candles,
                indicators=indicators,
                sentiment=sentiment,
                market_regime=market_regime,
                day_classification=day_classification,
                patterns=patterns,
            )
            prompts[target] = prompt
            if save_to_file:
                self._save_prompt(prompt, symbol, now, target)

        if prompts:
            self._last_prompt = prompts[self.targets[0]]
            self._last_prompt_time = now
            self._last_bundle = None
        return prompts

    def _build_prompt_bundle_artifact(
        self,
        symbol: str,
        timeframe_contexts: Dict[str, PromptTimeframeContext],
        primary_timeframe: str,
        sentiment: Dict[str, Any] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        day_classification: str = "unknown",
        patterns: List[Dict[str, Any]] = None,
        save_to_file: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> PromptBundleArtifact:
        timestamp = timestamp or datetime.now(timezone.utc)
        prompts: Dict[str, str] = {}
        bundle_payload = {
            "symbol": symbol,
            "generated_at": timestamp.isoformat(),
            "primary_timeframe": primary_timeframe,
            "timeframes": {
                tf: {
                    "timeframe": ctx.timeframe,
                    "candles": [asdict(candle) for candle in ctx.candles],
                    "indicators": [asdict(ind) for ind in ctx.indicators],
                    "sentiment": ctx.sentiment or {},
                    "market_regime": ctx.market_regime.value if hasattr(ctx.market_regime, "value") else str(ctx.market_regime),
                    "day_classification": ctx.day_classification,
                    "patterns": self._normalize_patterns(ctx.patterns),
                    "note": ctx.note,
                }
                for tf, ctx in timeframe_contexts.items()
            },
            "shared": {
                "sentiment": sentiment or {},
                "market_regime": market_regime.value if hasattr(market_regime, "value") else str(market_regime),
                "day_classification": day_classification,
                "patterns": self._normalize_patterns(patterns),
            },
        }

        for target in self.targets:
            prompt = self._build_multi_timeframe_prompt(
                target_name=target,
                symbol=symbol,
                timeframe_contexts=timeframe_contexts,
                primary_timeframe=primary_timeframe,
                sentiment=sentiment,
                market_regime=market_regime,
                day_classification=day_classification,
                patterns=patterns,
            )
            prompts[target] = prompt
            if save_to_file:
                self._save_prompt(prompt, symbol, timestamp, target)

        self._ensure_output_dir()
        latest_bundle_txt = self.output_dir / "latest_prompt_bundle.txt"
        latest_bundle_json = self.output_dir / "latest_prompt_bundle.json"
        archive_bundle_dir = self.output_dir / "archive"
        archive_bundle_dir.mkdir(exist_ok=True)
        archive_bundle_path = archive_bundle_dir / f"prompt_bundle_{symbol}_{timestamp.strftime('%Y%m%d_%H%M')}.json"
        if save_to_file:
            bundle_text = self._render_bundle_text(bundle_payload, prompts)
            latest_bundle_txt.write_text(bundle_text, encoding="utf-8")
            latest_bundle_json.write_text(json.dumps(bundle_payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
            archive_bundle_path.write_text(json.dumps(bundle_payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

        latest_prompt_path = self.output_dir / f"latest_prompt_{self.targets[0]}.txt"
        return PromptBundleArtifact(
            symbol=symbol,
            generated_at=timestamp,
            primary_timeframe=primary_timeframe,
            timeframe_contexts=timeframe_contexts,
            prompts=prompts,
            latest_prompt_path=latest_prompt_path,
            latest_bundle_txt_path=latest_bundle_txt,
            latest_bundle_json_path=latest_bundle_json,
            archive_bundle_path=archive_bundle_path,
        )

    def _render_bundle_text(self, bundle_payload: Dict[str, Any], prompts: Dict[str, str]) -> str:
        lines = [
            f"# Generated: {bundle_payload['generated_at']}",
            f"# Symbol: {bundle_payload['symbol']}",
            f"# Primary Timeframe: {bundle_payload['primary_timeframe']}",
            "# Included Timeframes: " + ", ".join(bundle_payload["timeframes"].keys()),
            "",
            "## Shared Context",
            json.dumps(bundle_payload["shared"], indent=2, ensure_ascii=True, default=str),
            "",
            "## Per-Target Prompts",
        ]
        for target, prompt in prompts.items():
            lines.append(f"### {target.upper()}")
            lines.append(prompt)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _save_prompt(self, prompt: str, symbol: str, timestamp: datetime, target_name: str):
        """Save prompt to file for easy access."""
        self._ensure_output_dir()
        target = str(target_name).lower()

        latest_path = self.output_dir / f"latest_prompt_{target}.txt"
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(f"# Generated: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n")
            f.write(f"# Symbol: {symbol}\n")
            f.write(f"# Target: {target.upper()}\n")
            f.write(f"# Copy everything below this line to {target.upper()}:\n")
            f.write("=" * 80 + "\n\n")
            f.write(prompt)

        # Backward compatible "latest_prompt.txt" mirrors the first target file.
        if target == self.targets[0]:
            generic_latest = self.output_dir / "latest_prompt.txt"
            with open(generic_latest, "w", encoding="utf-8") as f:
                f.write(f"# Generated: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n")
                f.write(f"# Symbol: {symbol}\n")
                f.write(f"# Target: {target.upper()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(prompt)

        archive_dir = self.output_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / f"prompt_{target}_{symbol}_{timestamp.strftime('%Y%m%d_%H%M')}.txt"
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        logger.info("Prompt saved to: %s", latest_path)

    def get_last_bundle(self) -> Optional[Dict[str, Any]]:
        """Get the last generated multi-timeframe bundle payload, if any."""
        return self._last_bundle

    def parse_response(self, response_text: str, target_name: str = "claude") -> Optional[AnalysisResult]:
        """
        Parse the AI response JSON into an AnalysisResult.

        Args:
            response_text: The JSON response from Claude Code or Codex

        Returns:
            AnalysisResult if parsing successful, None otherwise
        """
        try:
            # Clean up response - handle markdown code blocks
            json_text = response_text.strip()
            if json_text.startswith("```"):
                json_text = json_text.split("```")[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            data = json.loads(json_text)

            result = AnalysisResult(
                timestamp=datetime.now(timezone.utc),
                market_sentiment=data.get("market_sentiment", "neutral"),
                confidence=float(data.get("confidence", 0.5)),
                key_observations=data.get("key_observations", []),
                recommended_action=data.get("recommended_action", "WAIT"),
                risk_level=data.get("risk_level", "moderate"),
                reasoning=data.get("reasoning", ""),
            )

            self._last_result = result
            self._save_result(result, target_name=target_name)

            return result

        except json.JSONDecodeError as e:
            logger.error("Failed to parse response as JSON: %s", e)
            return None
        except Exception as e:
            logger.error("Error parsing response: %s", e)
            return None

    def _save_result(self, result: AnalysisResult, target_name: str = "claude"):
        """Save analysis result to file."""
        self._ensure_output_dir()

        result_path = self.output_dir / "latest_analysis.json"
        with open(result_path, "w", encoding="utf-8") as f:
            data = asdict(result)
            data["timestamp"] = result.timestamp.isoformat()
            json.dump(data, f, indent=2)

        target_result_path = self.output_dir / f"latest_analysis_{str(target_name).lower()}.json"
        with open(target_result_path, "w", encoding="utf-8") as f:
            data = asdict(result)
            data["timestamp"] = result.timestamp.isoformat()
            json.dump(data, f, indent=2)

        logger.info("Analysis result saved to: %s", result_path)

    def get_last_result(self) -> Optional[AnalysisResult]:
        """Get the last parsed analysis result."""
        return self._last_result

    def load_last_result(self, target_name: str = "") -> Optional[AnalysisResult]:
        """Load the last saved analysis result from file."""
        if target_name:
            result_path = self.output_dir / f"latest_analysis_{str(target_name).lower()}.json"
            if not result_path.exists():
                result_path = self.output_dir / "latest_analysis.json"
        else:
            result_path = self.output_dir / "latest_analysis.json"
        if not result_path.exists():
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                return AnalysisResult(**data)
        except Exception as e:
            logger.error("Failed to load analysis result: %s", e)
            return None

    def should_refresh(self, interval_minutes: int = 60) -> bool:
        """Check if a new prompt should be generated based on time interval."""
        if self._last_prompt_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_prompt_time).total_seconds()
        return elapsed >= interval_minutes * 60


# Convenience function for quick prompt generation
def generate_trading_prompt(
    symbol: str,
    candles: List[Candle],
    indicators: List[IndicatorResult],
    sentiment: Dict[str, Any] = None,
    market_regime: MarketRegime = MarketRegime.UNKNOWN,
    day_classification: str = "unknown",
    patterns: List[Dict[str, Any]] = None,
) -> str:
    """
    Quick helper to generate a trading analysis prompt.

    Example usage:
        prompt = generate_trading_prompt("BTCUSDT", candles, indicators)
        print(prompt)  # Copy this to Claude Code
    """
    generator = PromptGenerator()
    return generator.generate_prompt(
        symbol=symbol,
        candles=candles,
        indicators=indicators,
        sentiment=sentiment,
        market_regime=market_regime,
        day_classification=day_classification,
        patterns=patterns,
    )
