import pickle
import numpy as np


class Gmm_train(object):
    def __init__(self,clf,scaler):
        super(Gmm_train,self).__init__()
        self.clf = clf
        self.scaler = scaler

    def _normalize(self, features, train=False):
        """
        The features in the input 2D array are normalized.
        The rows are samples, the columns are features. If train==True then 
        the scaler is trained, else the trained scaler is used for the normalization.

        **Parameters:**

        ``features`` : 2D :py:class:`numpy.ndarray`
            Array of features to be normalized.

        **Returns:**

        ``features_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized array of features.

        """

        if self.scaler is not None:
            if train:
                self.scaler.fit(features)

            features_norm = self.scaler.transform(features)
        else:
            features_norm=features.copy()

        return features_norm
    def train_clf(self,features):
        real = self._normalize(features,train=True)
        X=real.copy()

        # Y=np.ones(len(real))

        self.clf.fit(X)
        return True
    def train_projector(self, training_features, projector_file):
        self.train_clf(training_features)

        # Save the GENERIC machine and normalizers:
        self.save_clf_and_mean_std(projector_file)
    def save_clf_and_mean_std(self, projector_file):
        """
        Saves the GENERIC machine, scaling parameters to a '.obj' file.
        The absolute name of the file is specified in ``projector_file`` string.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            Absolute name of the file to save the data to, as returned by
            ``bob.pad.base`` framework.

        ``machine`` : object
            The GENERIC machine to be saved. As returned by sklearn
            modules.


        """

        # Dumping Machine
        projector_file_n = projector_file[:-5]+'_skmodel.obj'
        with open(projector_file_n, 'wb') as fp:
            pickle.dump(self.clf, fp)

        # Dumping the scaler

        scaler_file_n = projector_file[:-5]+'_scaler.obj'
        with open(scaler_file_n, 'wb') as fp:
            pickle.dump(self.scaler, fp)
    def load_clf_and_mean_std(self, projector_file):
        """
        Loads the machine, features mean and std from the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            Absolute name of the file to load the trained projector from, as
            returned by ``bob.pad.base`` framework.

        **Returns:**

        ``machine`` : object
            The loaded GENERIC machine. As returned by sklearn.linear_model module.

        """

        projector_file_n = projector_file[:-5]+'_skmodel.obj'

        # Load the params of the machine:

        with open(projector_file_n, 'rb') as fp:
            self.clf = pickle.load(fp)

        scaler_file_n = projector_file[:-5]+'_scaler.obj'

        # Load parameters of the scaler:

        with open(scaler_file_n, 'rb') as fp:
            self.scaler = pickle.load(fp)


    #==========================================================================
    def load_projector(self, projector_file):
        """
        Loads the machine, features mean and std from the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        This function sets the arguments ``self.clf``, with loaded machines.


        Please register `performs_projection = True` in the constructor to
        enable this function.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            The file to read the projector from, as returned by the
            ``bob.pad.base`` framework. In this class the names of the files to
            read the projectors from are modified, see ``load_machine`` and
            ``load_cascade_of_machines`` methods of this class for more details.
        """

        self.load_clf_and_mean_std(projector_file)



    #==========================================================================
    def project(self, feature):
        """
        This function computes a vector of scores for each sample in the input
        array of features. The following steps are apllied:

        1. First, the input data is mean-std normalized using mean and std of the
           real class only.

        2. The input features are next classified using pre-trained GENERIC machine.

        Set ``performs_projection = True`` in the constructor to enable this function.
        It is assured that the :py:meth:`load_projector` was **called before** the
        ``project`` function is executed.

        **Parameters:**

        ``feature`` : FrameContainer or 2D :py:class:`numpy.ndarray`
            Two types of inputs are accepted.
            A Frame Container conteining the features of an individual,
            see ``bob.bio.video.utils.FrameContainer``.
            Or a 2D feature array of the size (N_samples x N_features).

        **Returns:**

        ``scores`` : 1D :py:class:`numpy.ndarray`
            Vector of scores. Scores for the real class are expected to be
            higher, than the scores of the negative / attack class.
            In this case scores are probabilities.
        """

        # 1. Convert input array to numpy array if necessary.
        # if isinstance(feature, FrameContainer):  # if FrameContainer convert to 2D numpy array

        #     features_array = convert_frame_cont_to_array(feature)

        # else:

        features_array = feature.copy()

        features_array_norm = self._normalize(features_array, train =False)

        # print(self.clf, dir(self.clf))
        # if self.one_class:
        scores= self.clf.predict_proba(features_array_norm)[:, 1]# this is available in new version -self.clf.mahalanobis(features_array_norm)  #
        return scores