class MSE:
    @staticmethod
    def mean_squared_error_grad(y_pred, y, h_grad):
        """
        h_grad - gradient of hypothesis function
        """
        return 2 * np.substract(y_pred, y).mean() * h_grad

    @staticmethod
    def mean_squared_error(y_pred, y):
        return np.square(np.substract(y_pred, y)).mean() 
        