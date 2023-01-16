import functools

from .models import casapose as cp
from .models import pose_models as pm
from .models import resnet as rn


class ModelsFactory:
    _models = {
        # Backbones
        "resnet18": rn.ResNet18,
        # 'resnet34': rn.ResNet34,
        # 'resnet50': rn.ResNet50,
        # 'resnet101': rn.ResNet101,
        # 'resnet152': rn.ResNet152,
        # Paper
        "casapose_c": pm.CASAPoseConditional1,  # BMVC (C) # casapose_cond_weighted_dtwin
        "casapose_c_gu": pm.CASAPoseConditional2,  # BMVC  (C/GU: 0x partial conv, guided us) # casapose_cond_weighted_combined_split_decoder_9
        "casapose_c_gcu3": pm.CASAPoseConditional3,  # BMVC (C/GCU3: 3x partial conv, guided us) # casapose_cond_weighted_combined_split_decoder_8
        "casapose_c_gcu4": pm.CASAPoseConditional4,  # BMVC (C/GCU4: 4x partial conv, guided us) # casapose_cond_weighted_combined_split_decoder_4
        "casapose_c_gcu5": pm.CASAPoseConditional5,  # BMVC (C/GCU5: 5x partial conv, guided us) # casapose_cond_weighted_combined_split_decoder_5
        "pvnet_combined": pm.PVNet,  # BMVC (Base)
        # Custom
        "casapose_custom": cp.CASAPoseConditional,  # like casapose_c_gcu5 but easy to reconfigure to other custom variants without code dublication (change decoder_params)
        # Alternative Models
        "casapose_c_gcu5_sw5": pm.CASAPoseConditional6,  # uses the same convolution weights in both decoders # casapose_cond_weighted_combined
        "casapose_c_gcu4_sw1": pm.CASAPoseConditional7,  # both decoders share initial convolution, 4x partial conv # casapose_cond_weighted_combined_split_decoder_2
        "casapose_c_gcu5_sw1": pm.CASAPoseConditional8,  # both decoders share initial convolution, 5x partial conv # casapose_cond_weighted_combined_split_decoder_3
        "casapose_c_gcu4_bilat": pm.CASAPoseConditional9,  # use guided bilateral upsampling, (4x partial conv, guided us)
        "casapose_c_gcu4_sw2": pm.CASAPoseConditional10,  # shared decoder for first two blocks decoder (4x partial conv, guided us)
        "pvnet": pm.PVNet,  # pvnet without merged output
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    @staticmethod
    def get_kwargs():
        return {}

    def inject_submodules(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            modules_kwargs = self.get_kwargs()
            new_kwargs = dict(list(kwargs.items()) + list(modules_kwargs.items()))
            return func(*args, **new_kwargs)

        return wrapper

    def get(self, name):
        if name not in self.models_names():
            raise ValueError("No such model `{}`, available models: {}".format(name, list(self.models_names())))

        model_fn = self.models[name]
        model_fn = self.inject_submodules(model_fn)
        return model_fn
