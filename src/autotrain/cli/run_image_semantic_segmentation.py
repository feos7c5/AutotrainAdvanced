from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.image_semantic_segmentation.params import ImageSemanticSegmentationParams

from . import BaseAutoTrainCommand


def run_image_semantic_segmentation_command_factory(args):
    return RunAutoTrainImageSemanticSegmentationCommand(args)


class RunAutoTrainImageSemanticSegmentationCommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(ImageSemanticSegmentationParams)
        arg_list = [
            {
                "arg": "--train",
                "help": "Command to train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Command to deploy the model (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Command to run inference (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--backend",
                "help": "Backend",
                "required": False,
                "type": str,
                "default": "local",
            },
        ] + arg_list
        run_image_semantic_segmentation_parser = parser.add_parser(
            "image-semantic-segmentation", description="✨ Run AutoTrain Image Semantic Segmentation"
        )
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if len(names) == 1:
                names = [arg["arg"]]

            kwargs = {
                "dest": arg["arg"].replace("--", "").replace("-", "_"),
                "help": arg["help"],
                "required": arg.get("required", False),
                "default": arg.get("default"),
                "choices": arg.get("choices"),
                "action": arg.get("action"),
            }
            
            if arg.get("action") != "store_true":
                kwargs["type"] = arg.get("type", str)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            run_image_semantic_segmentation_parser.add_argument(*names, **kwargs)

        run_image_semantic_segmentation_parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="Optional config file path to override parameters.",
        )

        run_image_semantic_segmentation_parser.add_argument(
            "--col-mapping",
            type=str,
            required=False,
            help="Optional column mapping for the dataset.",
        )

        run_image_semantic_segmentation_parser.set_defaults(func=run_image_semantic_segmentation_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "auto_find_batch_size",
            "push_to_hub",
            "ignore_mismatched_sizes",
            "reduce_labels",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

        if self.args.train:
            if self.args.project_name is None:
                raise ValueError("Project name must be specified")
            if self.args.data_path is None:
                raise ValueError("Data path must be specified")
            if self.args.model is None:
                raise ValueError("Model must be specified")
            if self.args.push_to_hub:
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

        if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
            if not self.args.push_to_hub:
                raise ValueError("Push to hub must be specified for spaces backend")
            if self.args.username is None:
                raise ValueError("Username must be specified for spaces backend")
            if self.args.token is None:
                raise ValueError("Token must be specified for spaces backend")

    def run(self):
        logger.info("Running Image Semantic Segmentation")
        if self.args.train:
            params = ImageSemanticSegmentationParams(**vars(self.args))
            project = AutoTrainProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}") 