"""
Human Graders - Manual grading interface with SQLite storage.

Provides:
- CLI interface for grading responses
- SQLite storage for human grades
- Comparison with model grades for calibration
- Export functionality for analysis

Usage:
    # Grade responses interactively
    python -m src.graders.human_graders grade

    # Compare human vs model grades
    python -m src.graders.human_graders compare

    # Export grades to JSON
    python -m src.graders.human_graders export
"""

import sqlite3
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path("eval_runs/human_grades.db")
DEFAULT_RESPONSES_PATH = Path("eval_runs/responses_to_grade.json")


class GradeLevel(Enum):
    """Human grade levels with descriptions."""
    EXCELLENT = (5, "Fully correct, comprehensive, well-formatted")
    GOOD = (4, "Mostly correct with minor issues")
    ACCEPTABLE = (3, "Partially correct, missing some information")
    POOR = (2, "Has relevant info but significant errors")
    FAIL = (1, "Wrong, irrelevant, or harmful")


@dataclass
class HumanGrade:
    """A human-provided grade for a response."""
    query_hash: str  # Hash of query for deduplication
    query: str
    response: str
    expected_agent: str
    actual_agent: str
    ground_truth: str

    # Human evaluation
    grade_level: int  # 1-5
    grade_label: str  # EXCELLENT, GOOD, etc.
    reasoning: str  # Why this grade
    grader_id: str  # Who graded (for multi-grader scenarios)

    # Metadata
    graded_at: str  # ISO timestamp
    eval_run_id: Optional[str] = None


@dataclass
class HumanGradeStats:
    """Statistics about human grading."""
    total_graded: int
    average_grade: float
    grade_distribution: Dict[str, int]
    by_agent: Dict[str, Dict[str, float]]  # agent -> {avg, count}
    agreement_with_model: Optional[float] = None


class HumanGraderStore:
    """SQLite storage for human grades."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS human_grades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    expected_agent TEXT,
                    actual_agent TEXT,
                    ground_truth TEXT,
                    grade_level INTEGER NOT NULL,
                    grade_label TEXT NOT NULL,
                    reasoning TEXT,
                    grader_id TEXT NOT NULL,
                    graded_at TEXT NOT NULL,
                    eval_run_id TEXT,
                    UNIQUE(query_hash, grader_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_grades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL UNIQUE,
                    query TEXT NOT NULL,
                    correctness_score REAL,
                    relevancy_score REAL,
                    recall_score REAL,
                    eval_run_id TEXT,
                    graded_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_hash
                ON human_grades(query_hash)
            """)

            conn.commit()

    @staticmethod
    def hash_query(query: str) -> str:
        """Generate a hash for a query for deduplication."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

    def save_human_grade(self, grade: HumanGrade) -> bool:
        """
        Save a human grade.

        Returns:
            True if saved, False if duplicate exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO human_grades
                    (query_hash, query, response, expected_agent, actual_agent,
                     ground_truth, grade_level, grade_label, reasoning,
                     grader_id, graded_at, eval_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    grade.query_hash, grade.query, grade.response,
                    grade.expected_agent, grade.actual_agent,
                    grade.ground_truth, grade.grade_level, grade.grade_label,
                    grade.reasoning, grade.grader_id, grade.graded_at,
                    grade.eval_run_id
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Grade already exists for query hash {grade.query_hash}")
            return False

    def save_model_grade(
        self,
        query: str,
        correctness: float,
        relevancy: float,
        recall: float,
        eval_run_id: Optional[str] = None
    ):
        """Save a model grade for comparison."""
        query_hash = self.hash_query(query)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_grades
                (query_hash, query, correctness_score, relevancy_score,
                 recall_score, eval_run_id, graded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_hash, query, correctness, relevancy, recall,
                eval_run_id, datetime.now().isoformat()
            ))
            conn.commit()

    def get_model_grade(self, query: str) -> Optional[Dict[str, float]]:
        """
        Get model grade scores for a query.

        Returns:
            Dict with 'correctness', 'relevancy', 'recall' scores, or None if not found.
        """
        query_hash = self.hash_query(query)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT correctness_score, relevancy_score, recall_score
                FROM model_grades
                WHERE query_hash = ?
            """, (query_hash,))
            row = cursor.fetchone()

            if row:
                return {
                    'correctness': row['correctness_score'],
                    'relevancy': row['relevancy_score'],
                    'recall': row['recall_score']
                }
        return None

    def get_human_grade(self, query: str, grader_id: str = "default") -> Optional[HumanGrade]:
        """Get a human grade for a query."""
        query_hash = self.hash_query(query)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM human_grades
                WHERE query_hash = ? AND grader_id = ?
            """, (query_hash, grader_id))
            row = cursor.fetchone()

            if row:
                return HumanGrade(
                    query_hash=row['query_hash'],
                    query=row['query'],
                    response=row['response'],
                    expected_agent=row['expected_agent'],
                    actual_agent=row['actual_agent'],
                    ground_truth=row['ground_truth'],
                    grade_level=row['grade_level'],
                    grade_label=row['grade_label'],
                    reasoning=row['reasoning'],
                    grader_id=row['grader_id'],
                    graded_at=row['graded_at'],
                    eval_run_id=row['eval_run_id']
                )
        return None

    def get_all_human_grades(self, grader_id: Optional[str] = None) -> List[HumanGrade]:
        """Get all human grades, optionally filtered by grader."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if grader_id:
                cursor = conn.execute(
                    "SELECT * FROM human_grades WHERE grader_id = ?",
                    (grader_id,)
                )
            else:
                cursor = conn.execute("SELECT * FROM human_grades")

            grades = []
            for row in cursor:
                grades.append(HumanGrade(
                    query_hash=row['query_hash'],
                    query=row['query'],
                    response=row['response'],
                    expected_agent=row['expected_agent'],
                    actual_agent=row['actual_agent'],
                    ground_truth=row['ground_truth'],
                    grade_level=row['grade_level'],
                    grade_label=row['grade_label'],
                    reasoning=row['reasoning'],
                    grader_id=row['grader_id'],
                    graded_at=row['graded_at'],
                    eval_run_id=row['eval_run_id']
                ))
            return grades

    def get_stats(self, grader_id: Optional[str] = None) -> HumanGradeStats:
        """Get grading statistics."""
        grades = self.get_all_human_grades(grader_id)

        if not grades:
            return HumanGradeStats(
                total_graded=0,
                average_grade=0.0,
                grade_distribution={},
                by_agent={}
            )

        # Grade distribution
        distribution = {}
        for level in GradeLevel:
            distribution[level.name] = sum(
                1 for g in grades if g.grade_level == level.value[0]
            )

        # By agent stats
        by_agent = {}
        for agent in ['structured', 'summary', 'needle']:
            agent_grades = [g for g in grades if g.actual_agent == agent]
            if agent_grades:
                avg = sum(g.grade_level for g in agent_grades) / len(agent_grades)
                by_agent[agent] = {'average': avg, 'count': len(agent_grades)}

        return HumanGradeStats(
            total_graded=len(grades),
            average_grade=sum(g.grade_level for g in grades) / len(grades),
            grade_distribution=distribution,
            by_agent=by_agent
        )

    def compare_with_model(self, grader_id: str = "default") -> List[Dict[str, Any]]:
        """Compare human grades with model grades."""
        comparisons = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT h.*, m.correctness_score, m.relevancy_score, m.recall_score
                FROM human_grades h
                LEFT JOIN model_grades m ON h.query_hash = m.query_hash
                WHERE h.grader_id = ?
            """, (grader_id,))

            for row in cursor:
                # Convert human grade (1-5) to 0-1 scale
                human_score = (row['grade_level'] - 1) / 4.0

                model_correctness = row['correctness_score']
                if model_correctness is not None:
                    diff = abs(human_score - model_correctness)
                    agreement = 1.0 - diff
                else:
                    agreement = None

                comparisons.append({
                    'query': row['query'][:50] + '...',
                    'human_grade': row['grade_level'],
                    'human_score': human_score,
                    'model_correctness': model_correctness,
                    'agreement': agreement,
                    'human_reasoning': row['reasoning']
                })

        return comparisons

    def export_to_json(self, output_path: Path) -> int:
        """Export all grades to JSON."""
        grades = self.get_all_human_grades()
        data = [asdict(g) for g in grades]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return len(data)

    def clear_all_grades(self) -> Tuple[int, int]:
        """
        Clear all human grades and model grades from the database.

        Called at the start of a new evaluation run to ensure
        grades match the current responses.

        Returns:
            Tuple of (human_grades_deleted, model_grades_deleted)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM human_grades")
            human_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM model_grades")
            model_count = cursor.fetchone()[0]

            conn.execute("DELETE FROM human_grades")
            conn.execute("DELETE FROM model_grades")
            conn.commit()

        return human_count, model_count

    def get_grade_counts(self) -> Tuple[int, int]:
        """Get count of existing grades."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM human_grades")
            human_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM model_grades")
            model_count = cursor.fetchone()[0]

        return human_count, model_count


class HumanGraderCLI:
    """Interactive CLI for human grading."""

    def __init__(
        self,
        store: HumanGraderStore,
        grader_id: str = "default"
    ):
        self.store = store
        self.grader_id = grader_id

    def grade_response(
        self,
        query: str,
        response: str,
        expected_agent: str,
        actual_agent: str,
        ground_truth: str,
        eval_run_id: Optional[str] = None
    ) -> Optional[HumanGrade]:
        """
        Interactive grading of a single response.

        Returns:
            HumanGrade if graded, None if skipped
        """
        # Check if already graded
        existing = self.store.get_human_grade(query, self.grader_id)
        if existing:
            print(f"Already graded (grade: {existing.grade_label}). Skip? [Y/n] ", end="")
            if input().strip().lower() != 'n':
                return existing

        # Display the response to grade
        print("\n" + "=" * 70)
        print("QUERY:", query)
        print("-" * 70)
        print("EXPECTED AGENT:", expected_agent)
        print("ACTUAL AGENT:", actual_agent)
        print("-" * 70)
        print("GROUND TRUTH:", ground_truth)
        print("-" * 70)
        print("RESPONSE:")
        print(response[:1000] + ("..." if len(response) > 1000 else ""))
        print("=" * 70)

        # Show grade options
        print("\nGrade options:")
        for level in GradeLevel:
            print(f"  {level.value[0]}. {level.name}: {level.value[1]}")
        print("  s. Skip this response")
        print("  q. Quit grading")

        # Get grade
        while True:
            choice = input("\nEnter grade (1-5, s, q): ").strip().lower()

            if choice == 'q':
                return None
            if choice == 's':
                print("Skipped.")
                return None

            try:
                grade_num = int(choice)
                if 1 <= grade_num <= 5:
                    break
                print("Invalid grade. Enter 1-5.")
            except ValueError:
                print("Invalid input.")

        # Find the grade level
        grade_level = None
        for level in GradeLevel:
            if level.value[0] == grade_num:
                grade_level = level
                break

        # Get reasoning
        reasoning = input("Brief reasoning (optional): ").strip()

        # Create and save grade
        grade = HumanGrade(
            query_hash=self.store.hash_query(query),
            query=query,
            response=response,
            expected_agent=expected_agent,
            actual_agent=actual_agent,
            ground_truth=ground_truth,
            grade_level=grade_num,
            grade_label=grade_level.name,
            reasoning=reasoning,
            grader_id=self.grader_id,
            graded_at=datetime.now().isoformat(),
            eval_run_id=eval_run_id
        )

        if self.store.save_human_grade(grade):
            print(f"Saved grade: {grade_level.name}")
        else:
            print("Grade already exists (not overwritten)")

        return grade

    def grade_batch(self, responses_file: Path) -> int:
        """
        Grade a batch of responses from a JSON file.

        Returns:
            Number of responses graded
        """
        if not responses_file.exists():
            print(f"File not found: {responses_file}")
            return 0

        with open(responses_file) as f:
            responses = json.load(f)

        print(f"\nLoaded {len(responses)} responses to grade")
        print(f"Grader ID: {self.grader_id}")
        print("-" * 40)

        graded = 0
        for i, resp in enumerate(responses, 1):
            print(f"\n[{i}/{len(responses)}]")

            result = self.grade_response(
                query=resp['query'],
                response=resp['response'],
                expected_agent=resp.get('expected_agent', 'unknown'),
                actual_agent=resp.get('actual_agent', 'unknown'),
                ground_truth=resp.get('ground_truth', ''),
                eval_run_id=resp.get('eval_run_id')
            )

            if result is None:
                # User quit
                print("\nGrading session ended.")
                break
            graded += 1

        return graded


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Human grading interface")
    parser.add_argument(
        'command',
        choices=['grade', 'compare', 'export', 'stats'],
        help='Command to run'
    )
    parser.add_argument(
        '--db',
        default=str(DEFAULT_DB_PATH),
        help='Path to grades database'
    )
    parser.add_argument(
        '--responses',
        default=str(DEFAULT_RESPONSES_PATH),
        help='Path to responses JSON file'
    )
    parser.add_argument(
        '--grader-id',
        default='default',
        help='Grader identifier'
    )
    parser.add_argument(
        '--output',
        default='eval_runs/grades_export.json',
        help='Output path for export'
    )

    args = parser.parse_args()

    store = HumanGraderStore(Path(args.db))

    if args.command == 'grade':
        cli = HumanGraderCLI(store, args.grader_id)
        graded = cli.grade_batch(Path(args.responses))
        print(f"\nTotal graded: {graded}")

    elif args.command == 'compare':
        comparisons = store.compare_with_model(args.grader_id)
        if not comparisons:
            print("No grades to compare")
            return

        print("\nHuman vs Model Grade Comparison")
        print("=" * 70)

        total_agreement = 0
        count = 0
        for c in comparisons:
            print(f"Query: {c['query']}")
            print(f"  Human: {c['human_grade']}/5 ({c['human_score']:.2f})")
            if c['model_correctness'] is not None:
                print(f"  Model: {c['model_correctness']:.2f}")
                print(f"  Agreement: {c['agreement']:.2f}")
                total_agreement += c['agreement']
                count += 1
            else:
                print("  Model: N/A")
            print()

        if count > 0:
            print(f"Average Agreement: {total_agreement/count:.2%}")

    elif args.command == 'export':
        count = store.export_to_json(Path(args.output))
        print(f"Exported {count} grades to {args.output}")

    elif args.command == 'stats':
        stats = store.get_stats(args.grader_id)
        print("\nHuman Grading Statistics")
        print("=" * 40)
        print(f"Total graded: {stats.total_graded}")
        print(f"Average grade: {stats.average_grade:.2f}/5")
        print("\nGrade distribution:")
        for label, count in stats.grade_distribution.items():
            print(f"  {label}: {count}")
        print("\nBy agent:")
        for agent, data in stats.by_agent.items():
            print(f"  {agent}: {data['average']:.2f} avg ({data['count']} graded)")


if __name__ == "__main__":
    main()
