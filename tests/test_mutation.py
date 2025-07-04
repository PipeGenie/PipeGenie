import pytest
from unittest.mock import MagicMock
from pipegenie.ea.gp import MultiMutation, HyperparameterMutation, BranchMutation, NodeMutation, NonTerminalNode, TerminalNode, Individual
from pipegenie.ea.gp._encoding import Primitive

class TestHyperparameterMutation:
    @pytest.fixture
    def setup(self):
        schema = MagicMock()

        mock_algorithm_a = MagicMock(return_value='algorithmA')
        mock_algorithm_a.name = 'algorithmA'
        mock_algorithm_a.arity = 1 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_a.__class__ = Primitive

        mock_hparam_1 = MagicMock(return_value='value')
        mock_hparam_1.name = 'value'
        mock_hparam_1.__class__ = type('HP1', (object,), {'name': 'value'})

        mock_hparam_2 = MagicMock(return_value='new_value')
        mock_hparam_2.name = 'new_value'
        mock_hparam_2.__class__ = type('HP2', (object,), {'name': 'new_value'})

        schema.terminals_map = {'algorithmA::param1': TerminalNode('algorithmA::param1', mock_hparam_2)}
        schema.get_parent_symbols = MagicMock(return_value=['classifier'])
        
        ind = Individual([NonTerminalNode('classifier', ''), TerminalNode('algorithmA', mock_algorithm_a), NonTerminalNode('classifier_hp', ''), TerminalNode('algorithmA::param1', mock_hparam_1)])
        
        return schema, ind
    
    def test_no_mutation(self, setup):
        schema, ind = setup
        mutation = HyperparameterMutation(0) # 0% chance of mutation

        son, changed = mutation.mutate(ind, schema)
        
        for son_node, parent_node in zip(son, ind):
            assert son_node.symbol == parent_node.symbol

        assert not changed

    def test_hyperparameter_mutation(self, setup):
        schema, ind = setup
        mutation = HyperparameterMutation(1) # 100% chance of mutation
        son = ind.clone() # TODO: clone ind inside operation?

        son, changed = mutation.mutate(son, schema)
        
        assert ind[0] == son[0]
        assert ind[1] == son[1]
        assert ind[2] == son[2]
        assert ind[3].symbol == 'algorithmA::param1'
        assert ind[3].code() == 'value'
        assert son[3].symbol == 'algorithmA::param1'
        assert son[3].code() == 'new_value'
        assert changed

class TestBranchMutation:
    @pytest.fixture
    def setup(self):
        schema = MagicMock()

        def mock_fill_tree_branch(tree, symbol, number_of_derivs):
            """
            Mock implementation of fill_tree_branch for testing.
            """
            tree.append(schema.non_terminals_map[symbol])
            prod_symbol = schema.non_terminals_map[symbol].production

            if prod_symbol in schema.terminals_map:
                tree.append(schema.terminals_map[prod_symbol])

        schema.max_deriv_size = 10
        schema.min_derivations = MagicMock(return_value=1)
        schema.fill_tree_branch = MagicMock(side_effect=mock_fill_tree_branch)

        mock_algorithm_a = MagicMock(return_value='algorithmA')
        mock_algorithm_a.name = 'algorithmA'
        mock_algorithm_a.arity = 0 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_a.__class__ = Primitive

        mock_algorithm_b = MagicMock(return_value='algorithmB')
        mock_algorithm_b.name = 'algorithmB'
        mock_algorithm_b.arity = 0 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_b.__class__ = Primitive

        schema.non_terminals_map = {'classifier': NonTerminalNode('classifier', 'algorithmB')}
        schema.terminals_map = {'algorithmB': TerminalNode('algorithmB', mock_algorithm_b)}
        
        ind = Individual([NonTerminalNode('classifier', ''), TerminalNode('algorithmA', mock_algorithm_a)])
        
        return schema, ind
    
    def test_no_mutation(self, setup):
        schema, ind = setup
        mutation = BranchMutation(0) # 0% chance of mutation

        son, changed = mutation.mutate(ind, schema)
        
        for son_node, parent_node in zip(son, ind):
            assert son_node.symbol == parent_node.symbol

        assert not changed

    def test_branch_mutation(self, setup):
        schema, ind = setup
        mutation = BranchMutation(1) # 100% chance of mutation
        son = ind.clone() # TODO: clone ind inside operation?

        son, changed = mutation.mutate(son, schema)
        
        assert ind[0].symbol == son[0].symbol
        assert ind[1].symbol == 'algorithmA'
        assert son[1].symbol == 'algorithmB'
        assert changed

class TestNodeMutation:
    @pytest.fixture
    def setup(self):
        schema = MagicMock()

        def mock_fill_tree_branch(tree, symbol, number_of_derivs):
            """
            Mock implementation of fill_tree_branch for testing.
            """
            tree.append(schema.non_terminals_map[symbol])
            prod_symbol = schema.non_terminals_map[symbol].production

            if prod_symbol in schema.terminals_map:
                tree.append(schema.terminals_map[prod_symbol])

        schema.max_deriv_size = 10
        schema.min_derivations = MagicMock(return_value=1)
        schema.max_derivations = MagicMock(return_value=1)
        schema.fill_tree_branch = MagicMock(side_effect=mock_fill_tree_branch)
        schema.get_parent_symbols = MagicMock(return_value=['classifier'])

        mock_algorithm_a = MagicMock(return_value='algorithmA')
        mock_algorithm_a.name = 'algorithmA'
        mock_algorithm_a.arity = 1 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_a.__class__ = Primitive

        mock_hparam_1 = MagicMock(return_value='value')
        mock_hparam_1.name = 'value'
        mock_hparam_1.__class__ = type('HP1', (object,), {'name': 'value'})

        mock_algorithm_b = MagicMock(return_value='algorithmB')
        mock_algorithm_b.name = 'algorithmB'
        mock_algorithm_b.arity = 1 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_b.__class__ = Primitive

        mock_hparam_2 = MagicMock(return_value='new_value')
        mock_hparam_2.name = 'new_value'
        mock_hparam_2.__class__ = type('HP2', (object,), {'name': 'new_value'})

        schema.terminals_map = {'algorithmA::param1': TerminalNode('algorithmA::param1', mock_hparam_2), 'algorithmB': TerminalNode('algorithmB', mock_algorithm_b)}
        schema.non_terminals_map = {'classifier': NonTerminalNode('classifier', 'algorithmB')}
        
        ind = Individual([NonTerminalNode('classifier', 'algorithmA;classifier_hp'), TerminalNode('algorithmA', mock_algorithm_a), NonTerminalNode('classifier_hp', 'algorithmA::param1'), TerminalNode('algorithmA::param1', mock_hparam_1)])
        
        return schema, ind
    
    def test_no_mutation(self, setup):
        schema, ind = setup
        mutation = NodeMutation(0) # 0% chance of mutation

        son, changed = mutation.mutate(ind, schema)
        
        for son_node, parent_node in zip(son, ind):
            assert son_node.symbol == parent_node.symbol

        assert not changed

    def test_node_mutation_hyperparameter(self, setup):
        schema, ind = setup
        mutation = NodeMutation(1, 1) # 100% chance of mutation and 100% chance of changing a hyperparameter
        son = ind.clone() # TODO: clone ind inside operation?

        son, changed = mutation.mutate(son, schema)
        
        assert ind[0].symbol == son[0].symbol
        assert ind[1].symbol == son[1].symbol
        assert ind[2].symbol == son[2].symbol
        assert ind[3].symbol == 'algorithmA::param1'
        assert ind[3].code() == 'value'
        assert son[3].symbol == 'algorithmA::param1'
        assert son[3].code() == 'new_value'
        assert changed

    def test_node_mutation_algorithm(self, setup):
        schema, ind = setup
        mutation = NodeMutation(1, 0) # 100% chance of mutation and 0% chance of changing a hyperparameter
        son = ind.clone()

        son, changed = mutation.mutate(son, schema)

        assert ind[0].symbol == son[0].symbol
        assert ind[1].symbol == 'algorithmA'
        assert son[1].symbol == 'algorithmB'
        assert changed

class TestMultiMutation:
    @pytest.fixture
    def setup(self):
        schema = MagicMock()

        def mock_fill_tree_branch(tree, symbol, number_of_derivs):
            """
            Mock implementation of fill_tree_branch for testing.
            """
            tree.append(schema.non_terminals_map[symbol])
            prod_symbol = schema.non_terminals_map[symbol].production

            if prod_symbol in schema.terminals_map:
                tree.append(schema.terminals_map[prod_symbol])

        schema.max_deriv_size = 10
        schema.min_derivations = MagicMock(return_value=1)
        schema.fill_tree_branch = MagicMock(side_effect=mock_fill_tree_branch)
        schema.get_parent_symbols = MagicMock(return_value=['classifier'])

        mock_algorithm_a = MagicMock(return_value='algorithmA')
        mock_algorithm_a.name = 'algorithmA'
        mock_algorithm_a.arity = 1 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_a.__class__ = Primitive

        mock_hparam_1 = MagicMock(return_value='value')
        mock_hparam_1.name = 'value'
        mock_hparam_1.__class__ = type('HP1', (object,), {'name': 'value'})

        mock_algorithm_b = MagicMock(return_value='algorithmB')
        mock_algorithm_b.name = 'algorithmB'
        mock_algorithm_b.arity = 1 + 1 # Add an unexistent parameter to ensure arity > 0
        mock_algorithm_b.__class__ = Primitive

        mock_hparam_2 = MagicMock(return_value='new_value')
        mock_hparam_2.name = 'new_value'
        mock_hparam_2.__class__ = type('HP2', (object,), {'name': 'new_value'})

        schema.terminals_map = {'algorithmA::param1': TerminalNode('algorithmA::param1', mock_hparam_2), 'algorithmB': TerminalNode('algorithmB', mock_algorithm_b)}
        schema.non_terminals_map = {'classifier': NonTerminalNode('classifier', 'algorithmB')}
        
        ind = Individual([NonTerminalNode('classifier', 'algorithmA;classifier_hp'), TerminalNode('algorithmA', mock_algorithm_a), NonTerminalNode('classifier_hp', 'algorithmA::param1'), TerminalNode('algorithmA::param1', mock_hparam_1)])

        return schema, ind
    
    def test_no_mutation(self, setup):
        schema, ind = setup
        mutation = MultiMutation(0) # 0% chance of mutation

        son, changed = mutation.mutate(ind, schema)
        
        for son_node, parent_node in zip(son, ind):
            assert son_node.symbol == parent_node.symbol

        assert not changed

    def test_multi_mutation_hyperparameter(self, setup):
        schema, ind = setup
        mutation = MultiMutation(1, 1) # 100% chance of mutation and 100% chance of changing a hyperparameter
        son = ind.clone()

        son, changed = mutation.mutate(son, schema)

        assert ind[0].symbol == son[0].symbol
        assert ind[1].symbol == son[1].symbol
        assert ind[2].symbol == son[2].symbol
        assert ind[3].symbol == 'algorithmA::param1'
        assert ind[3].code() == 'value'
        assert son[3].symbol == 'algorithmA::param1'
        assert son[3].code() == 'new_value'
        assert changed

    def test_multi_mutation_algorithm(self, setup):
        schema, ind = setup
        mutation = MultiMutation(1, 0) # 100% chance of mutation and 0% chance of changing a hyperparameter
        son = ind.clone()

        son, changed = mutation.mutate(son, schema)

        assert ind[0].symbol == son[0].symbol
        assert ind[1].symbol == 'algorithmA'
        assert son[1].symbol == 'algorithmB'
        assert changed