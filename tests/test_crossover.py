import pytest
from unittest.mock import patch, MagicMock
from pipegenie.ea.gp import BranchCrossover, MultiCrossover, NonTerminalNode, TerminalNode, Individual
from pipegenie.ea.gp._encoding import Primitive

class TestBranchCrossover:
    @pytest.fixture
    def setup(self):
        schema = MagicMock()
        schema.max_deriv_size = 10
        schema.root = 'workflow'

        mock_prep_algorithm_a = MagicMock(return_value='prep_algorithmA')
        mock_prep_algorithm_a.name = 'prep_algorithmA'
        mock_prep_algorithm_a.arity = 0 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_prep_algorithm_a.__class__ = Primitive

        mock_prep_algorithm_b = MagicMock(return_value='prep_algorithmB')
        mock_prep_algorithm_b.name = 'prep_algorithmB'
        mock_prep_algorithm_b.arity = 0 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_prep_algorithm_b.__class__ = Primitive

        mock_cls_algorithm_a = MagicMock(return_value='cls_algorithmA')
        mock_cls_algorithm_a.name = 'cls_algorithmA'
        mock_cls_algorithm_a.arity = 0 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_cls_algorithm_a.__class__ = Primitive

        mock_cls_algorithm_b = MagicMock(return_value='cls_algorithmB')
        mock_cls_algorithm_b.name = 'cls_algorithmB'
        mock_cls_algorithm_b.arity = 0 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_cls_algorithm_b.__class__ = Primitive

        # Two individual, with only one common non-terminal node (preprocessA)
        ind1 = Individual([NonTerminalNode('preprocessA', ''), TerminalNode('prep_algorithmA', mock_prep_algorithm_a), NonTerminalNode('preprocessB', ''), NonTerminalNode('classifierA', ''), TerminalNode('cls_algorithmA', mock_cls_algorithm_a)])
        ind2 = Individual([NonTerminalNode('preprocessA', ''), TerminalNode('prep_algorithmB', mock_prep_algorithm_b), NonTerminalNode('preprocessC', ''), NonTerminalNode('classifierB', ''), TerminalNode('cls_algorithmB', mock_cls_algorithm_b)])

        return schema, ind1, ind2, mock_prep_algorithm_a, mock_cls_algorithm_a, mock_prep_algorithm_b, mock_cls_algorithm_b
    
    def test_no_crossover(self, setup):
        schema, ind1, ind2, _, _, _, _ = setup
        crossover = BranchCrossover(0) # 0% chance of crossover

        son1, son2, changed = crossover.cross(ind1, ind2, schema)
        
        for son_node, parent_node in zip(son1, ind1):
            assert son_node.symbol == parent_node.symbol

        for son_node, parent_node in zip(son2, ind2):
            assert son_node.symbol == parent_node.symbol

        assert not changed

    def test_no_common_non_terminal(self, setup):
        schema, ind1, _, _, mock_prep_algorithm_b, _, mock_cls_algorithm_b = setup

        # Change the common non-terminal node (preprocessA) to a different one (preprocessD) so there is no common non-terminal node
        ind2 = Individual([NonTerminalNode('preprocessD', ''), TerminalNode('prep_algorithmB', mock_prep_algorithm_b), NonTerminalNode('preprocessC', ''), NonTerminalNode('classifierB', ''), TerminalNode('cls_algorithmB', mock_cls_algorithm_b)])

        crossover = BranchCrossover(1) # 100% chance of crossover

        with patch('random.randint') as randint:
            randint.return_value = 0 # Start checking from the first node
            son1, son2, changed = crossover.cross(ind1, ind2, schema)
            
            for son_node, parent_node in zip(son1, ind1):
                assert son_node.symbol == parent_node.symbol

            for son_node, parent_node in zip(son2, ind2):
                assert son_node.symbol == parent_node.symbol
                
            assert not changed

    def test_swap_branches(self, setup):
        schema, ind1, ind2, mock_prep_algorithm_a, mock_cls_algorithm_a, mock_prep_algorithm_b, mock_cls_algorithm_b = setup
        crossover = BranchCrossover(1)

        with patch('random.randint') as randint:
            randint.return_value = 0 # Start checking from the first node
            son1, son2, changed = crossover.cross(ind1, ind2, schema)
            # prep_algorithmA was in ind1 and prep_algorithmB in ind2, so after the crossover, prep_algorithmB should be in ind1 and mock_prep_algorithm_a in ind2            
            new_ind1 = Individual([NonTerminalNode('preprocessA', ''), TerminalNode('prep_algorithmB', mock_prep_algorithm_b), NonTerminalNode('preprocessB', ''), NonTerminalNode('classifierA', ''), TerminalNode('cls_algorithmA', mock_cls_algorithm_a)])
            new_ind2 = Individual([NonTerminalNode('preprocessA', ''), TerminalNode('prep_algorithmA', mock_prep_algorithm_a), NonTerminalNode('preprocessC', ''), NonTerminalNode('classifierB', ''), TerminalNode('cls_algorithmB', mock_cls_algorithm_b)])

            for son_node, parent_node in zip(son1, new_ind1):
                assert son_node.symbol == parent_node.symbol

            for son_node, parent_node in zip(son2, new_ind2):
                assert son_node.symbol == parent_node.symbol

            assert changed

class TestMultiCrossover:
    @pytest.fixture
    def setup(self):
        schema = MagicMock()
        schema.max_deriv_size = 10
        schema.root = 'workflow'

        mock_algoritm_a = MagicMock(return_value='algorithmA')
        mock_algoritm_a.name = 'algorithmA'
        mock_algoritm_a.arity = 2 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algoritm_a.__class__ = Primitive

        mock_hparamA_1_1 = MagicMock(return_value=0)
        mock_hparamA_1_1.name = '0'
        mock_hparamA_1_2 = MagicMock(return_value=True)
        mock_hparamA_1_2.name = 'True'

        mock_hparamA_2_1 = MagicMock(return_value=5)
        mock_hparamA_2_1.name = '5'
        mock_hparamA_2_2 = MagicMock(return_value=False)
        mock_hparamA_2_2.name = 'False'

        mock_algoritm_b = MagicMock(return_value='algorithmB')
        mock_algoritm_b.name = 'algorithmB'
        mock_algoritm_b.arity = 2 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algoritm_b.__class__ = Primitive

        mock_hparamB_1 = MagicMock(return_value=3)
        mock_hparamB_1.name = '3'
        mock_hparamB_2 = MagicMock(return_value='balanced')
        mock_hparamB_2.name = 'balanced'

        ind1 = Individual([NonTerminalNode('classifier', 'algorithmA;classifier_hp'), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', 'p1;p2'), TerminalNode('algorithmA::param1', mock_hparamA_1_1), TerminalNode('algorithmA::param2', mock_hparamA_1_2)])
        ind2 = Individual([NonTerminalNode('classifier', 'algorithmA;classifier_hp'), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', 'p1;p2'), TerminalNode('algorithmA::param1', mock_hparamA_2_1), TerminalNode('algorithmA::param2', mock_hparamA_2_2)])

        return schema, ind1, ind2, mock_algoritm_a, mock_hparamA_1_1, mock_hparamA_1_2, mock_hparamA_2_1, mock_hparamA_2_2, mock_algoritm_b, mock_hparamB_1, mock_hparamB_2
    
    def test_no_crossover(self, setup):
        schema, ind1, ind2, _, _, _, _, _, _, _, _ = setup
        crossover = MultiCrossover(0) # 0% chance of crossover

        son1, son2, changed = crossover.cross(ind1, ind2, schema)
        
        for son_node, parent_node in zip(son1, ind1):
            assert son_node.symbol == parent_node.symbol

        for son_node, parent_node in zip(son2, ind2):
            assert son_node.symbol == parent_node.symbol

        assert not changed

    def test_no_common_non_terminal(self, setup):
        schema, ind1, _, mock_algoritm_a, _, _, mock_hparamA_2_1, mock_hparamA_2_2, _, _, _ = setup

        # Change the common non-terminal node (classifier) to a different one (regressor) so there is no common non-terminal node
        ind2 = Individual([NonTerminalNode('regressor', ''), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('regressor_hp', ''), TerminalNode('algorithmA::param1', mock_hparamA_2_1), TerminalNode('algorithmA::param2', mock_hparamA_2_2)])
        crossover = MultiCrossover(1) # 100% chance of crossover

        with patch('random.randint') as randint:
            randint.return_value = 0 # Start checking from the first node
            son1, son2, changed = crossover.cross(ind1, ind2, schema)
            
            for son_node, parent_node in zip(son1, ind1):
                assert son_node.symbol == parent_node.symbol

            for son_node, parent_node in zip(son2, ind2):
                assert son_node.symbol == parent_node.symbol

            assert not changed

    def test_swap_hyperparameters_all(self, setup):
        schema, ind1, ind2, mock_algoritm_a, mock_hparamA_1_1, mock_hparamA_1_2, mock_hparamA_2_1, mock_hparamA_2_2, _, _, _ = setup
        crossover = MultiCrossover(1) # 100% chance of crossover

        with patch('random.randint') as randint:
            randint.return_value = 2 # Swap the 2 common hyperparameters
            son1, son2, changed = crossover.cross(ind1, ind2, schema)
            # algorithmA is in common, so the hyperparameters should be swapped (2 in this case because of random.randint)
            new_ind1 = Individual([NonTerminalNode('classifier', ''), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', ''), TerminalNode('algorithmA::param1', mock_hparamA_2_1), TerminalNode('algorithmA::param2', mock_hparamA_2_2)])
            new_ind2 = Individual([NonTerminalNode('classifier', ''), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', ''), TerminalNode('algorithmA::param1', mock_hparamA_1_1), TerminalNode('algorithmA::param2', mock_hparamA_1_2)])
            
            for son_node, parent_node in zip(son1, new_ind1):
                assert son_node.symbol == parent_node.symbol

            for son_node, parent_node in zip(son2, new_ind2):
                assert son_node.symbol == parent_node.symbol

            assert changed

    def test_swap_hyperparameters_one(self, setup):
        schema, ind1, ind2, mock_algoritm_a, mock_hparamA_1_1, mock_hparamA_1_2, mock_hparamA_2_1, mock_hparamA_2_2, _, _, _ = setup
        crossover = MultiCrossover(1) # 100% chance of crossover

        with patch('random.randint') as randint:
            randint.return_value = 1 # Swap the first common hyperparameter
            son1, son2, changed = crossover.cross(ind1, ind2, schema)
            # algorithmA is in common, so the hyperparameters should be swapped (1 in this case because of random.randint)
            new_ind1 = Individual([NonTerminalNode('classifier', ''), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', ''), TerminalNode('algorithmA::param1', mock_hparamA_2_1), TerminalNode('algorithmA::param2', mock_hparamA_1_2)])
            new_ind2 = Individual([NonTerminalNode('classifier', ''), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', ''), TerminalNode('algorithmA::param1', mock_hparamA_1_1), TerminalNode('algorithmA::param2', mock_hparamA_2_2)])
            
            for son_node, parent_node in zip(son1, new_ind1):
                assert son_node.symbol == parent_node.symbol

            for son_node, parent_node in zip(son2, new_ind2):
                assert son_node.symbol == parent_node.symbol

            assert changed

    def test_swap_branches(self, setup):
        schema, ind1, _, mock_algoritm_a, mock_hparamA_1_1, mock_hparamA_1_2, _, _, mock_algoritm_b, mock_hparamB_1, mock_hparamB_2 = setup
        crossover = MultiCrossover(1) # 100% chance of crossover

        # Change the common algorithm (algorithmA) to a different one (algorithmB) so there algorithms are swapped instead of hyperparameters
        ind2 = Individual([NonTerminalNode('classifier', 'algorithmB;classifier_hp'), TerminalNode('algorithmB', mock_algoritm_b), NonTerminalNode('classifier_hp', 'p1;p2'), TerminalNode('algorithmB::param1', mock_hparamB_1), TerminalNode('algorithmB::param2', mock_hparamB_2)])

        with patch('random.randint') as randint:
            randint.return_value = 0
            son1, son2, changed = crossover.cross(ind1, ind2, schema)
            # No common hyperparameters, so the branches should be swapped
            new_ind1 = Individual([NonTerminalNode('classifier', 'algorithmB;classifier_hp'), TerminalNode('algorithmB', mock_algoritm_b), NonTerminalNode('classifier_hp', 'p1;p2'), TerminalNode('algorithmB::param1', mock_hparamB_1), TerminalNode('algorithmB::param2', mock_hparamB_2)])
            new_ind2 = Individual([NonTerminalNode('classifier', 'algorithmA;classifier_hp'), TerminalNode('algorithmA', mock_algoritm_a), NonTerminalNode('classifier_hp', 'p1;p2'), TerminalNode('algorithmA::param1', mock_hparamA_1_1), TerminalNode('algorithmA::param2', mock_hparamA_1_2)])
            
            for son_node, parent_node in zip(son1, new_ind1):
                assert son_node.symbol == parent_node.symbol

            for son_node, parent_node in zip(son2, new_ind2):
                assert son_node.symbol == parent_node.symbol

            assert changed