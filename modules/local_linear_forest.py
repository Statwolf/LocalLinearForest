import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

class LocalLinearForestRegressor(RandomForestRegressor):
    def __init__(self,
                 n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False)

        # matrix with leaf node for training observation
        #  in each tree
        self._incidence_matrix = None
        self._X_train = None
        self._Y_train = None

    def _extract_leaf_nodes_ids(self, X):
        '''
        Extract a matrix of dimension (rows, cols), where rows is the number of rows of X and cols is the number of tree in the forest,
        \nthat contains the ids of the leaf node for each observation in each tree.

        Parameters:
        -----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            \n``dtype=np.float32``. If a sparse matrix is provided, it will be
            \nconverted into a sparse ``csr_matrix``.
        '''
        leafs = []
        for e in self.estimators_:
            leafs.append(e.apply(X).reshape(-1, 1))

        leaf_nodes_ids = np.concatenate(leafs, axis=1)

        # the number of the rows must be the same of the number of observation
        assert leaf_nodes_ids.shape[0] == X.shape[0]
        # the number of the columns must be the same of the number of estimators (trees)
        assert leaf_nodes_ids.shape[1] == len(self.estimators_)

        return leaf_nodes_ids


    def fit(self, X, y, sample_weight=None):
        '''
        Override 
        '''
        super().fit(X, y, sample_weight=sample_weight)
        # save train data
        self._X_train = X
        self._Y_train = y
        # calculate leaf nodes for each observation in each tree
        self._incidence_matrix = self._extract_leaf_nodes_ids(X)
        return self

    def _get_forest_coefficients(self, observation_leaf_ids):
        '''
        1   B   {1 | Xi € Lb(X_actual)}
        -  sum -------------------------
        B  b=1       |Lb(X_actual)|

        Parameters:
        -----------
        observation_leaf_ids: numpy.array [1, n_estimators_]
        '''
        coeffs = []
        for i in range(0, self._X_train.shape[0]):
            count = 0
            for j in range(0, observation_leaf_ids.shape[1]):
                if self._incidence_matrix[i, j] == observation_leaf_ids[0, j]:
                    count += 1 / (self._incidence_matrix[:, j] == observation_leaf_ids[0, j]).sum()
            coeffs.append(1 / self.n_estimators * count)
        return coeffs
    
    def predict(self, X):
        '''
        Override
        '''
        results = []

        X_ = np.array(X)

        # prediction for each observation
        for i in range(0, X_.shape[0]):
            X_actual = X_[0, :].reshape(1, -1)
            # we can calulate the coefficients for one row at a time
            actual_leaf_ids = self._extract_leaf_nodes_ids(X_actual)
            # calculate coefficients weights alpha_i(X_actual)
            alphas = self._get_forest_coefficients(actual_leaf_ids)
            # X_i - X_actual
            X_disc = self._X_train - X_actual
            # ridge
            ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_disc, self._Y_train, alphas)
            # ridge predictions
            results.append(ridge.predict(X_actual))

        return results
        
    