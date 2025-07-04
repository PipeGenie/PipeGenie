import pytest
from pipegenie.ea.gp.tools import Statistics, MultiStatistics

class TestStatistics():
    """
    Test the Statistics class.
    """
    def test_register(self):
        """
        Test the register method of the Statistics class.
        """
        statistics = Statistics()
        statistics.register('mean', lambda x: sum(x) / len(x))
        statistics.register('std', lambda x: (sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)) ** 0.5)
        statistics.register('max', max)
        statistics.register('min', min)
        
        assert len(statistics.functions) == 4
        assert 'mean' in statistics.functions
        assert 'std' in statistics.functions
        assert 'max' in statistics.functions
        assert 'min' in statistics.functions

    def test_compile(self):
        """
        Test the compile method of the Statistics class.
        """
        statistics = Statistics()
        statistics.register('mean', lambda x: sum(x) / len(x))
        statistics.register('std', lambda x: (sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)) ** 0.5)
        statistics.register('max', max)
        statistics.register('min', min)
        
        data = [1, 2, 3, 4, 5]
        result = statistics.compile(data)
        
        assert result['mean'] == pytest.approx(3)
        assert result['std'] == pytest.approx(1.4142135623730951)
        assert result['max'] == 5
        assert result['min'] == 1

    def test_compile_key(self):
        """
        Test the compile method of the Statistics class with a key function to extract the values.
        """
        statistics = Statistics(key=lambda x: x[1])
        statistics.register('mean', lambda x: sum(x) / len(x))
        statistics.register('std', lambda x: (sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)) ** 0.5)
        statistics.register('max', max)
        statistics.register('min', min)
        
        data = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
        result = statistics.compile(data)
        
        assert result['mean'] == pytest.approx(3)
        assert result['std'] == pytest.approx(1.4142135623730951)
        assert result['max'] == 5
        assert result['min'] == 1

class TestMultiStatistics():
    """
    Test the MultiStatistics class.
    """
    def test_fields(self):
        """
        Test the fields attribute of the MultiStatistics class.
        """
        statistics_1 = Statistics()
        statistics_2 = Statistics()
        multi_statistics = MultiStatistics(stat_1=statistics_1, stat_2=statistics_2)

        assert 'stat_1' in multi_statistics
        assert 'stat_2' in multi_statistics
        # the fields attribute should be sorted
        assert multi_statistics.fields != ['stat_2', 'stat_1']
        assert multi_statistics.fields == ['stat_1', 'stat_2']

    def test_register(self):
        """
        Test the register method of the MultiStatistics class.
        """
        statistics_1 = Statistics()
        statistics_1.register('max', max) # this function should only be registered in statistics_1
        statistics_2 = Statistics()
        statistics_2.register('min', min) # this function should only be registered in statistics_2
        multi_statistics = MultiStatistics(stat_1=statistics_1, stat_2=statistics_2)

        # this functions should be registered in both statistics
        multi_statistics.register('mean', lambda x: sum(x) / len(x))
        multi_statistics.register('std', lambda x: (sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)) ** 0.5)
        
        assert len(statistics_1.functions) == 3
        assert 'mean' in statistics_1.functions
        assert 'std' in statistics_1.functions
        assert 'max' in statistics_1.functions
        assert 'min' not in statistics_1.functions
        
        assert len(statistics_2.functions) == 3
        assert 'mean' in statistics_2.functions
        assert 'std' in statistics_2.functions
        assert 'max' not in statistics_2.functions
        assert 'min' in statistics_2.functions

    def test_compile(self):
        """
        Test the compile method of the MultiStatistics class.
        """
        statistics_1 = Statistics()
        statistics_2 = Statistics()
        multi_statistics = MultiStatistics(stat_1=statistics_1, stat_2=statistics_2)

        multi_statistics.register('mean', lambda x: sum(x) / len(x))
        multi_statistics.register('std', lambda x: (sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)) ** 0.5)
        multi_statistics.register('max', max)
        multi_statistics.register('min', min)
        
        data = [1, 2, 3, 4, 5]
        result = multi_statistics.compile(data)
        
        assert result['stat_1']['mean'] == pytest.approx(3)
        assert result['stat_1']['std'] == pytest.approx(1.4142135623730951)
        assert result['stat_1']['max'] == 5
        assert result['stat_1']['min'] == 1
        
        assert result['stat_2']['mean'] == pytest.approx(3)
        assert result['stat_2']['std'] == pytest.approx(1.4142135623730951)
        assert result['stat_2']['max'] == 5
        assert result['stat_2']['min'] == 1

    def test_compile_key(self):
        """
        Test the compile method of the MultiStatistics class having Statistics objects with a different key function.
        """
        statistics_1 = Statistics(key=lambda x: x[0])
        statistics_2 = Statistics(key=lambda x: x[1])
        multi_statistics = MultiStatistics(stat_1=statistics_1, stat_2=statistics_2)

        multi_statistics.register('mean', lambda x: sum(x) / len(x))
        multi_statistics.register('std', lambda x: (sum((i - sum(x) / len(x)) ** 2 for i in x) / len(x)) ** 0.5)
        multi_statistics.register('max', max)
        multi_statistics.register('min', min)
        
        data = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
        result = multi_statistics.compile(data)

        assert result['stat_1']['mean'] == pytest.approx(1)
        assert result['stat_1']['std'] == pytest.approx(0)
        assert result['stat_1']['max'] == 1
        assert result['stat_1']['min'] == 1

        assert result['stat_2']['mean'] == pytest.approx(3)
        assert result['stat_2']['std'] == pytest.approx(1.4142135623730951)
        assert result['stat_2']['max'] == 5
        assert result['stat_2']['min'] == 1