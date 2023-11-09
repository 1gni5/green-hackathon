import tensorflow as tf


class Serializer:
    """
    Serialize a given TensorFlow model for use in a Machine Learning.
    """

    def __init__(self):
        pass

    def get_trainable_parameters(self, model):
        """
        Get the number of trainable parameters in a model.
        """
        return int(
            sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        )

    def get_non_trainable_parameters(self, model):
        """
        Get the number of non-trainable parameters in a model.
        """
        return int(
            sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        )

    def get_tf_layer_classes(self):
        """
        Get a list of all the layer class names in TensorFlow.

        Returns:
        - layer_classes: List of strings, each representing a layer class name.
        """
        layer_classes = [
            name
            for name in dir(tf.keras.layers)
            if isinstance(getattr(tf.keras.layers, name), type)
            and issubclass(getattr(tf.keras.layers, name), tf.keras.layers.Layer)
        ]
        return layer_classes

    def serialize(self, model, epochs, batch_size):
        """
        Serialize the model to a given path.
        """
        serialized_model = {
            "epochs": epochs,
            "batch_size": batch_size,
            "trainable_params": self.get_trainable_parameters(model),
            "non_trainable_params": self.get_non_trainable_parameters(model),
        }

        # All layer in tensorflow
        for layer in self.get_tf_layer_classes():
            serialized_model[layer] = 0

        for layer in model.layers:
            type_of_layer = layer.__class__.__name__
            serialized_model[type_of_layer] += 1

        # Add boolean if GPU is available
        serialized_model["gpu_available"] = tf.test.is_gpu_available()

        return serialized_model
