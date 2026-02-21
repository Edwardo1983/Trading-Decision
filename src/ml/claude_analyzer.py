from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import Candle, IndicatorResult, MarketRegime

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


class PromptGenerator:
    """
    Generates structured prompts for manual execution on Claude Code CLI or Codex.

    Usage:
        1. Call generate_prompt() to get the formatted prompt
        2. Copy the prompt to Claude Code CLI or Codex
        3. Paste the AI response back using parse_response()

    The prompt is saved to a file for easy access.
    """

    # Main analysis prompt template - can be customized
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

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        prompts: Dict[str, str] = {}
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
        return prompts

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
