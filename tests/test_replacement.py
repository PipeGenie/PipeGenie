import pytest
from unittest.mock import MagicMock, patch
from pipegenie.ea import ElitistGenerationalReplacement

class TestElitismGenerationalReplacement:
    @pytest.fixture
    def setup(self):
        with patch("pipegenie.ea.gp.Individual") as MockIndividual:
            with patch("pipegenie.ea.gp.Fitness") as MockFitness:
                # Ensure that each call to MockIndividual() returns a new MagicMock instance
                MockIndividual.side_effect = lambda: MagicMock()
                MockFitness.side_effect = lambda: MagicMock()

                def create_individual(valid=True, weighted_value=0):
                    fitness = MockFitness()
                    fitness.valid = valid
                    fitness.weighted_value = weighted_value
                    individual = MockIndividual()
                    individual.fitness = fitness
                    return individual
                
                population = [
                    create_individual(valid=True, weighted_value=5),
                    create_individual(valid=True, weighted_value=3),
                    create_individual(valid=False),  # Invalid individual
                    create_individual(valid=True, weighted_value=7)
                ]

                offspring = [
                    create_individual(valid=True, weighted_value=6),
                    create_individual(valid=False),  # Invalid individual
                    create_individual(valid=True, weighted_value=2),
                    create_individual(valid=True, weighted_value=8)
                ]

                elite = [
                    # Note that the elite individuals are sorted by weighted value
                    create_individual(valid=True, weighted_value=10),
                    create_individual(valid=True, weighted_value=9)
                ]

                yield population, offspring, elite
            
    def test_perform_replacement_no_elitism(self, setup):
        population, offspring, elite = setup
        print(population)
        print(offspring)
        print(elite)
        replacement = ElitistGenerationalReplacement(elitism=0)
        new_population = replacement.replace(population, offspring, elite)
        print(new_population[0].fitness.weighted_value)
        assert len(new_population) == 4
        # No elitism, so the offspring should replace the population
        # The offspring should be sorted by validity and then by weighted value
        assert new_population[0].fitness.valid == False
        assert new_population[1].fitness.weighted_value == 2
        assert new_population[2].fitness.weighted_value == 6
        assert new_population[3].fitness.weighted_value == 8

    def test_perform_replacement_1_elitism(self, setup):
        population, offspring, elite = setup
        replacement = ElitistGenerationalReplacement(elitism=1)
        new_population = replacement.replace(population, offspring, elite)
        assert len(new_population) == 4
        # Elitism is 1, so the best elite individual should replace the invalid individual
        assert new_population[0].fitness.weighted_value == 10
        assert new_population[1].fitness.weighted_value == 2
        assert new_population[2].fitness.weighted_value == 6
        assert new_population[3].fitness.weighted_value == 8

    def test_perform_replacement_2_elitism(self, setup):
        population, offspring, elite = setup
        replacement = ElitistGenerationalReplacement(elitism=2)
        new_population = replacement.replace(population, offspring, elite)
        assert len(new_population) == 4
        # Elitism is 2, so the whole elite should replace the invalid individual and the worst offspring individual
        assert new_population[0].fitness.weighted_value == 10
        assert new_population[1].fitness.weighted_value == 9
        assert new_population[2].fitness.weighted_value == 6
        assert new_population[3].fitness.weighted_value == 8

    def test_perform_replacement_3_elitism(self, setup):
        population, offspring, elite = setup
        replacement = ElitistGenerationalReplacement(elitism=3)
        new_population = replacement.replace(population, offspring, elite)
        assert len(new_population) == 4
        # Elitism is 3, but there are only 2 elite individuals, so the whole elite should replace the invalid individual and the worst offspring individual
        assert new_population[0].fitness.weighted_value == 10
        assert new_population[1].fitness.weighted_value == 9
        assert new_population[2].fitness.weighted_value == 6
        assert new_population[3].fitness.weighted_value == 8
