#!/usr/bin/env python3
"""
Agentic Quality Checker for Video Transcription Pipeline

Uses an LLM to evaluate transcript quality and decide:
- PASS: Transcript is good, use it
- RETRY: Issues detected, should retry with different settings
- MANUAL_REVIEW: Needs human review

Also includes rule-based pre-checks for obvious issues.
"""

import os
import re
from collections import Counter
from typing import Tuple, Dict
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai


class QualityVerdict(Enum):
    PASS = "pass"
    RETRY = "retry"
    MANUAL_REVIEW = "manual_review"


@dataclass
class QualityReport:
    verdict: QualityVerdict
    score: float  # 0-100
    issues: list
    recommendations: list
    details: str


class TranscriptQualityChecker:
    """
    Two-stage quality checker:
    1. Rule-based pre-checks (fast, catches obvious issues)
    2. LLM-based semantic check (catches subtle issues)
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model_name = model_name

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    def check_quality(self, transcript: str, use_llm: bool = True) -> QualityReport:
        """
        Run quality checks on transcript.

        Args:
            transcript: The transcript text to check
            use_llm: Whether to use LLM for semantic checking (slower but more thorough)

        Returns:
            QualityReport with verdict, score, and details
        """
        issues = []
        recommendations = []

        # Stage 1: Rule-based pre-checks
        rule_issues = self._rule_based_checks(transcript)
        issues.extend(rule_issues)

        # If rule-based checks find critical issues, don't bother with LLM
        critical_issues = [i for i in rule_issues if i.get('severity') == 'critical']
        if critical_issues:
            return QualityReport(
                verdict=QualityVerdict.RETRY,
                score=0,
                issues=[i['message'] for i in rule_issues],
                recommendations=["Retry transcription with different model or settings"],
                details=f"Critical issues found: {[i['message'] for i in critical_issues]}"
            )

        # Stage 2: LLM-based semantic check (if enabled and available)
        if use_llm and self.model:
            llm_report = self._llm_quality_check(transcript)
            issues.extend(llm_report.get('issues', []))
            recommendations.extend(llm_report.get('recommendations', []))

            # Combine scores
            rule_score = self._calculate_rule_score(rule_issues)
            llm_score = llm_report.get('score', 70)
            final_score = (rule_score * 0.4) + (llm_score * 0.6)
        else:
            rule_score = self._calculate_rule_score(rule_issues)
            final_score = rule_score

        # Determine verdict
        if final_score >= 70 and not any(i.get('severity') == 'high' for i in rule_issues):
            verdict = QualityVerdict.PASS
        elif final_score >= 40:
            verdict = QualityVerdict.RETRY
        else:
            verdict = QualityVerdict.MANUAL_REVIEW

        return QualityReport(
            verdict=verdict,
            score=final_score,
            issues=[i['message'] if isinstance(i, dict) else i for i in issues],
            recommendations=recommendations,
            details=f"Rule score: {rule_score:.0f}, Final score: {final_score:.0f}"
        )

    def _rule_based_checks(self, transcript: str) -> list:
        """Fast rule-based checks for obvious issues"""
        issues = []

        # Check 1: Excessive marker repetition ([noise], [inaudible], etc.)
        markers = re.findall(r'\[[^\]]+\]', transcript)
        if markers:
            marker_counts = Counter(markers)
            most_common_marker, count = marker_counts.most_common(1)[0]

            total_markers = sum(marker_counts.values())
            lines = [l for l in transcript.split('\n') if l.strip()]

            # If one marker appears >50 times or makes up >40% of lines, it's a problem
            if count > 50:
                issues.append({
                    'type': 'marker_spam',
                    'severity': 'critical',
                    'message': f"Marker '{most_common_marker}' repeated {count} times (hallucination)"
                })
            elif total_markers > len(lines) * 0.5:
                issues.append({
                    'type': 'excessive_markers',
                    'severity': 'high',
                    'message': f"Excessive markers: {total_markers} markers in {len(lines)} lines ({total_markers/len(lines)*100:.0f}%)"
                })

        # Check 2: Line repetition
        lines = [l.strip() for l in transcript.split('\n') if l.strip()]
        if len(lines) >= 5:
            # Remove timestamps for comparison
            content_lines = []
            for line in lines:
                # Remove timestamp prefix
                cleaned = re.sub(r'^\d{1,2}:\d{2}\s*', '', line)
                if cleaned:
                    content_lines.append(cleaned)

            if content_lines:
                line_counts = Counter(content_lines)
                most_common_line, count = line_counts.most_common(1)[0]

                if count > 5 and count > len(content_lines) * 0.15:
                    issues.append({
                        'type': 'line_repetition',
                        'severity': 'high',
                        'message': f"Line repeated {count} times: '{most_common_line[:50]}...'"
                    })

        # Check 3: Word-level repetition (same word >100 times)
        words = re.findall(r'\b[a-zA-Z]{2,}\b', transcript.lower())
        if words:
            word_counts = Counter(words)
            # Exclude common words
            common_words = {'the', 'and', 'is', 'it', 'to', 'of', 'in', 'that', 'you', 'for', 'on', 'are', 'with', 'this'}
            for word, count in word_counts.most_common(5):
                if word not in common_words and count > 100:
                    issues.append({
                        'type': 'word_spam',
                        'severity': 'critical',
                        'message': f"Word '{word}' repeated {count} times (hallucination)"
                    })
                    break

        # Check 4: Transcript too short
        if len(transcript.strip()) < 100:
            issues.append({
                'type': 'too_short',
                'severity': 'high',
                'message': f"Transcript too short: {len(transcript)} characters"
            })

        # Check 5: No timestamps
        timestamp_pattern = r'\d{1,2}:\d{2}'
        timestamps = re.findall(timestamp_pattern, transcript)
        if len(timestamps) < 3:
            issues.append({
                'type': 'no_timestamps',
                'severity': 'high',
                'message': f"Only {len(timestamps)} timestamps found"
            })

        # Check 6: Consecutive identical lines
        prev_line = None
        consecutive_count = 0
        max_consecutive = 0
        for line in lines:
            cleaned = re.sub(r'^\d{1,2}:\d{2}\s*', '', line).strip()
            if cleaned == prev_line and cleaned:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
            prev_line = cleaned

        if max_consecutive >= 2:  # 3+ consecutive identical lines
            issues.append({
                'type': 'consecutive_repetition',
                'severity': 'medium' if max_consecutive == 2 else 'high',
                'message': f"Same content repeated {max_consecutive + 1} times consecutively"
            })

        return issues

    def _calculate_rule_score(self, issues: list) -> float:
        """Calculate score based on rule-based issues"""
        score = 100

        for issue in issues:
            severity = issue.get('severity', 'low')
            if severity == 'critical':
                score -= 50
            elif severity == 'high':
                score -= 25
            elif severity == 'medium':
                score -= 10
            else:
                score -= 5

        return max(0, score)

    def _llm_quality_check(self, transcript: str) -> Dict:
        """Use LLM to check semantic quality of transcript"""

        # Truncate if too long
        if len(transcript) > 8000:
            transcript = transcript[:8000] + "\n[...truncated...]"

        prompt = f"""Analyze this classroom video transcript for quality issues.

TRANSCRIPT:
{transcript}

Evaluate the transcript and respond with a JSON object:
{{
    "score": <0-100 quality score>,
    "issues": [<list of specific issues found>],
    "recommendations": [<list of recommendations>],
    "is_coherent": <true/false - does dialogue make sense?>,
    "has_hallucinations": <true/false - obvious fabricated content?>
}}

Consider:
1. Does the dialogue sound like a real classroom?
2. Are speaker transitions natural?
3. Is there repetitive/hallucinated content?
4. Are timestamps reasonable and sequential?
5. Is the content coherent and meaningful?

Return ONLY the JSON object, no other text."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024}
            )

            # Parse response
            text = response.text.strip()

            # Try to extract JSON
            import json
            if text.startswith('{'):
                result = json.loads(text)
            else:
                # Try to find JSON in response
                json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {"score": 70, "issues": [], "recommendations": []}

            return result

        except Exception as e:
            print(f"LLM quality check error: {e}")
            return {"score": 70, "issues": [], "recommendations": []}


def check_transcript_quality(transcript: str, use_llm: bool = False) -> QualityReport:
    """
    Convenience function to check transcript quality.

    Args:
        transcript: The transcript text
        use_llm: Whether to use LLM for deeper analysis

    Returns:
        QualityReport with verdict and details
    """
    checker = TranscriptQualityChecker()
    return checker.check_quality(transcript, use_llm=use_llm)


# Quick test
if __name__ == "__main__":
    # Test with a bad transcript (lots of [noise])
    bad_transcript = """
00:00 Hello
00:02 [noise] [noise] [noise] [noise] [noise]
00:04 [noise] [noise] [noise] [noise] [noise]
00:06 [noise] [noise] [noise] [noise] [noise]
"""

    # Test with a good transcript
    good_transcript = """
00:00 Teacher: Good morning class!
00:05 Student1: Good morning!
00:08 Teacher: Today we're going to learn about area.
00:15 Student2: Like the area of a rectangle?
00:18 Teacher: Exactly! Can anyone tell me the formula?
"""

    print("=" * 60)
    print("Testing BAD transcript:")
    print("=" * 60)
    report = check_transcript_quality(bad_transcript)
    print(f"Verdict: {report.verdict.value}")
    print(f"Score: {report.score}")
    print(f"Issues: {report.issues}")
    print()

    print("=" * 60)
    print("Testing GOOD transcript:")
    print("=" * 60)
    report = check_transcript_quality(good_transcript)
    print(f"Verdict: {report.verdict.value}")
    print(f"Score: {report.score}")
    print(f"Issues: {report.issues}")
