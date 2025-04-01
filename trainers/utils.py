import random
import itertools
import numpy as np


def apply_strategy(self, responses, bb_scores):
    # Flatten responses and scores
    if isinstance(responses[0], list):
        completions = list(itertools.chain.from_iterable(responses))
        rewards = list(itertools.chain.from_iterable(bb_scores))
        batch_size = len(responses[0])
    else:
        completions = responses
        rewards = bb_scores
        batch_size = 0

    if self.strategy == "Oracle_Single":
        # Substitute a random guess with the target
        idx = random.randint(0, len(completions) - 1)
        completions[idx] = self.target
        rewards[idx] = 1.0
    elif self.strategy == "Online_Single":
        pass
    elif self.strategy == "Online_Mean":
        completions = responses
        rewards = [np.mean(scores) for scores in bb_scores]
    elif self.strategy == "Online_Max":
        completions = responses
        rewards = [np.max(scores) for scores in bb_scores]
    elif self.strategy == "Online_Batch_Mean":
        # Sort completions by scores and then batch them with batch's mean rewards
        word_scores = [[word, score] for word, score in zip(completions, rewards)]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + batch_size] for i in range(0, len(word_scores), batch_size)]
        completions = [[y[0] for y in x] for x in word_scores]
        rewards = [np.mean([y[1] for y in x]) for x in word_scores]
    elif self.strategy == "Online_Batch_Max":
        # Sort completions by scores and then batch them with batch's max rewards
        word_scores = [[word, score] for word, score in zip(completions, rewards)]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + batch_size] for i in range(0, len(word_scores), batch_size)]
        completions = [[y[0] for y in x] for x in word_scores]
        rewards = [np.max([y[1] for y in x]) for x in word_scores]
    elif self.strategy == "Greedy_Single":
        # Substitute a random guess with the best guess so far
        if len(self.past_guesses) > 0:
            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[0]
            idx = random.randint(0, len(completions) - 1)
            completions[idx] = best_guess[0]
            rewards[idx] = best_guess[1]
    elif self.strategy == "Greedy_Single_Related":
        # Substitute a random guess with the best guess so far
        if len(self.past_guesses) > 0:
            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[0]
            idx = random.randint(0, len(completions) - 1)
            completions[idx] = best_guess[0]
            rewards[idx] = best_guess[1]

        completion_rewards = sorted(list(zip(completions, rewards)), key=lambda x: x[1])
        related_completions, related_rewards = self.sample_arc_related_completions(
            inputs[0]["prompt"][-1]["content"], completion_rewards[-1][0]
        )
        completion_rewards[:5] = list(zip(related_completions, related_rewards))
        print("TEST", completion_rewards)

        completions = [x[0] for x in completion_rewards]
        rewards = [x[1] for x in completion_rewards]

    elif self.strategy == "Greedy_Batch_Mean":
        completions = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        word_scores = [[word, score] for word, score in zip(completions, scores)]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
        # Substitute a random guess batch with the best guess batch so far
        if len(self.past_guesses) > 0:
            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[:2]
            idx = random.randint(0, len(word_scores) - 1)
            word_scores[idx] = best_guess  # type:ignore
        completions = [[x[0][0], x[1][0]] for x in word_scores]
        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
    elif self.strategy == "Greedy_Batch_Max":
        completions = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        word_scores = [[word, score] for word, score in zip(completions, scores)]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
        # Substitute a random guess batch with the best guess batch so far
        if len(self.past_guesses) > 0:
            best_guess = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)[:2]
            idx = random.randint(0, len(word_scores) - 1)
            word_scores[idx] = best_guess  # type:ignore
        completions = [[x[0][0], x[1][0]] for x in word_scores]
        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
        # rewards = []
        # rewards.append([np.max([x[0][1], x[1][1]]) for x in word_scores])
        # rewards.append([int(x[0][0] != x[1][0]) for x in word_scores])
    elif self.strategy == "TopDelta_Batch_Mean":
        completions = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        word_scores = [[word, score] for word, score in zip(completions, scores)]
        random.shuffle(word_scores)
        if len(self.past_guesses) > 0:
            word_scores = word_scores[:6]
            past_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
            word_scores = word_scores + past_guesses[:2] + past_guesses[-2:]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
        completions = [[x[0][0], x[1][0]] for x in word_scores]
        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
    elif self.strategy == "TopDelta_Batch_Max":
        completions = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        word_scores = [[word, score] for word, score in zip(completions, scores)]
        random.shuffle(word_scores)
        if len(self.past_guesses) > 0:
            word_scores = word_scores[:6]
            past_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
            word_scores = word_scores + past_guesses[:2] + past_guesses[-2:]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
        completions = [[x[0][0], x[1][0]] for x in word_scores]
        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
    elif self.strategy == "Greedy_Batch_Mean_Related":
        completions = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        word_scores = [[word, score] for word, score in zip(completions, scores)]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
        # Substitute a random guess batch with the best guess batch so far
        best_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
        if len(self.past_guesses) > 0:
            random.shuffle(word_scores)
            word_scores[0] = best_guesses[:2]  # type:ignore
        completions = [[x[0][0], x[1][0]] for x in word_scores]
        rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]

        related_completions = None
        if len(self.past_guesses) > 0:
            related_completions = self.sample_related_completions(best_guesses[0][0], 5)
        if related_completions is not None:
            print("RELATED", related_completions)
            print("BEFORE", completions)
            print("BEFORE", rewards)
            random.shuffle(related_completions)
            related_completions = related_completions[:4]
            related_completions = [[x, self.get_bb_score(self.target, x)] for x in related_completions]
            idx = random.sample(range(1, 5), 3)
            # old_best = word_scores[0]
            word_scores = related_completions + word_scores[idx[0]] + word_scores[idx[1]] + word_scores[idx[2]]
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            # word_scores = old_best + word_scores
            word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
            completions = [[x[0][0], x[1][0]] for x in word_scores]
            rewards = [np.mean([x[0][1], x[1][1]]) for x in word_scores]
            print("AFTER", completions)
            print("AFTER", rewards)
    elif self.strategy == "Greedy_Batch_Max_Related":
        completions = list(itertools.chain.from_iterable(responses))
        scores = list(itertools.chain.from_iterable(bb_scores))
        word_scores = [[word, score] for word, score in zip(completions, scores)]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
        # Substitute a random guess batch with the best guess batch so far
        best_guesses = sorted(self.past_guesses.items(), key=lambda x: x[1], reverse=True)
        if len(self.past_guesses) > 0:
            random.shuffle(word_scores)
            word_scores[0] = best_guesses[:2]  # type:ignore
        completions = [[x[0][0], x[1][0]] for x in word_scores]
        rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]

        related_completions = None
        if len(self.past_guesses) > 0:
            related_completions = self.sample_related_completions(best_guesses[0][0], 5)
        if related_completions is not None:
            print("RELATED", related_completions)
            print("BEFORE", completions)
            print("BEFORE", rewards)
            random.shuffle(related_completions)
            related_completions = related_completions[:4]
            related_completions = [[x, self.get_bb_score(self.target, x)] for x in related_completions]
            idx = random.sample(range(1, 5), 3)
            # old_best = word_scores[0]
            word_scores = related_completions + word_scores[idx[0]] + word_scores[idx[1]] + word_scores[idx[2]]
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            # word_scores = old_best + word_scores
            word_scores = [word_scores[i : i + 2] for i in range(0, len(word_scores), 2)]
            completions = [[x[0][0], x[1][0]] for x in word_scores]
            rewards = [np.max([x[0][1], x[1][1]]) for x in word_scores]
            print("AFTER", completions)
            print("AFTER", rewards)

    return completions, rewards
