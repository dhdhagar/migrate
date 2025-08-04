from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import uuid
import random


@dataclass
class Solution:
    """Represents a solution in the database"""

    id: str
    completion: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solution":
        """Create from dictionary representation"""
        return cls(**data)


class EvolutionDatabase:
    """
    Database for tracking and sampling solutions
    """

    def __init__(
        self,
        num_islands: int,
        migration_interval: int,
        migration_rate: float,
        archive_size: int,
        exploration_ratio: float,
        exploitation_ratio: float,
        exploration_topk: int,
        exploitation_topk: int,
    ):
        self.solutions: Dict[str, Solution] = {}
        self.islands: List[Dict[str, Any]] = [{"by_id": set(), "by_completion": {}} for _ in range(num_islands)]
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.archive: Dict[str, Any] = {"by_id": set(), "by_completion": set()}
        self.archive_size = archive_size
        self.current_island: int = 0
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.exploration_topk = exploration_topk
        self.exploitation_topk = exploitation_topk

    def add(self, completion_score: Tuple[str, float], target_island: Optional[int] = None) -> str:
        if target_island == None:
            target_island = self.current_island
        completion, score = completion_score

        # Do not add if the completion already exists in the island
        if score == 0 or completion in self.islands[target_island]["by_completion"]:
            return ""

        id = str(uuid.uuid4())
        solution = Solution(id=id, completion=completion, score=score)
        self.solutions[id] = solution
        self.islands[target_island]["by_id"].add(id)
        self.islands[target_island]["by_completion"][completion] = id
        self.update_archive(solution)
        return id

    def update_archive(self, solution: Solution) -> None:
        """
        Update the archive of elite solutions
        """
        if len(self.archive["by_completion"]) < self.archive_size:
            self.archive["by_id"].add(solution.id)
            self.archive["by_completion"].add(solution.completion)
            return

        archive_solutions = [self.solutions[id] for id in self.archive["by_id"]]
        worst_solution = min(archive_solutions, key=lambda x: x.score)

        # Replace if new solution is better
        if solution.score > worst_solution.score:
            if solution.completion not in self.archive["by_completion"]:
                # If the completion-to-add is already in the archive, do not remove the worse
                self.archive["by_completion"].remove(worst_solution.completion)
                for x in list(self.archive["by_id"]):
                    if self.solutions[x].completion == worst_solution.completion:
                        self.archive["by_id"].remove(x)

            self.archive["by_id"].add(solution.id)
            self.archive["by_completion"].add(solution.completion)

    def sample(self) -> Any:
        empty_island = self.is_current_island_empty()
        rand_val = random.random()
        if not empty_island and rand_val < self.exploration_ratio:
            # EXPLORATION: Sample from current island (diverse sampling)
            current_island_solutions = self.islands[self.current_island]["by_id"]
            if not current_island_solutions:
                rand_id = random.choice(list(self.solutions.keys()))
                return [self.solutions[rand_id]]
            else:
                # Randomly sample from the top island scores not in the archive
                island_solutions = [id for id in current_island_solutions if id not in self.archive["by_id"]]
                if len(island_solutions) == 0:
                    island_solutions = [id for id in current_island_solutions]
                island_solutions.sort(key=lambda x: self.solutions[x].score, reverse=True)
                print(
                    "Picking from island (explore):",
                    [(self.solutions[x].completion, self.solutions[x].score) for x in island_solutions[:10]],
                )
                return [self.solutions[random.choice(island_solutions[: self.exploration_topk])]]
        elif not empty_island and rand_val < self.exploration_ratio + self.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite solutions)
            archive_solutions_in_island = [
                id for id in self.archive["by_id"] if id in self.islands[self.current_island]["by_id"]
            ]
            if archive_solutions_in_island:
                # If the island have solutions in the archive, then randomly sample top_k from the these island archive solutions
                archive_solutions_in_island = sorted(
                    archive_solutions_in_island, key=lambda x: self.solutions[x].score, reverse=True
                )
                print(
                    "Picking from island-archive (exploit):",
                    [(self.solutions[x].completion, self.solutions[x].score) for x in archive_solutions_in_island],
                )
                return [self.solutions[random.choice(archive_solutions_in_island)]]
            else:
                # If the island have no solutions in the archive, then randomly sample top_k from the island
                current_island_solutions = list(self.islands[self.current_island]["by_id"])
                current_island_solutions.sort(key=lambda x: self.solutions[x].score, reverse=True)
                print(
                    "Picking from island (exploit):",
                    [self.solutions[x].completion for x in current_island_solutions[:10]],
                )
                return [self.solutions[random.choice(current_island_solutions[: self.exploitation_topk])]]
        else:
            # RANDOM: Sample from any solution (remaining probability)
            rand_id = random.choice(list(self.solutions.keys()))
            return self.solutions[rand_id]

    def migrate(self) -> None:
        migrants_per_island = []
        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue

            island_solutions = [self.solutions[id] for id in island["by_id"]]
            island_solutions.sort(key=lambda x: x.score, reverse=True)
            num_to_migrate = max(1, int(len(island_solutions) * self.migration_rate))
            migrants = island_solutions[:num_to_migrate]
            migrants_per_island.append(migrants)

        for i, migrants in enumerate(migrants_per_island):
            # Migrate to adjacent islands (ring topology)
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]
            for migrant in migrants:
                for target_island in target_islands:
                    if migrant.completion in self.islands[target_island]["by_completion"]:
                        continue

                    id = str(uuid.uuid4())
                    migrant_copy = Solution(id=id, completion=migrant.completion, score=migrant.score)
                    self.solutions[id] = migrant_copy
                    self.islands[target_island]["by_id"].add(id)
                    self.islands[target_island]["by_completion"][migrant.completion] = id
                    if migrant.id in self.archive["by_id"]:
                        self.archive["by_id"].add(migrant_copy.id)

    def next_island(self) -> None:
        self.current_island = (self.current_island + 1) % len(self.islands)

    def is_current_island_empty(self) -> bool:
        return len(self.islands[self.current_island]["by_id"]) == 0
