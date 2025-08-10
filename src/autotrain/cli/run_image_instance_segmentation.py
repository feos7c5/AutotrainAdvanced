import argparse

from autotrain import logger
from autotrain.cli.utils import common_args
from autotrain.trainers.image_instance_segmentation.params import ImageInstanceSegmentationParams


def run_image_instance_segmentation_command_factory(args):
    return RunAutoTrainImageInstanceSegmentationCommand(args)


class RunAutoTrainImageInstanceSegmentationCommand:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        from autotrain.cli.run_image_instance_segmentation import add_subparser
        add_subparser(parser)

    def run(self):
        logger.info("Running Image Instance Segmentation")
        if self.args.train:
            params = ImageInstanceSegmentationParams(**vars(self.args))
            params.save(output_dir=self.args.project_name)
            if self.args.backend.startswith("spaces"):
                from autotrain.backend import SpaceRunner

                sr = SpaceRunner(
                    params=params,
                    backend=self.args.backend,
                )
                space_id = sr.prepare()
                print(f"Space created: {space_id}")
            else:
                from autotrain.trainers.image_instance_segmentation import train

                train(params)


def add_subparser(parser):
    from autotrain.trainers.image_instance_segmentation.params import ImageInstanceSegmentationParams
    from autotrain.cli.utils import get_field_info
    
    arg_list = get_field_info(ImageInstanceSegmentationParams)
    arg_list = [
        {
            "arg": "--train",
            "help": "Command to train the model",
            "required": False,
            "action": "store_true",
        },
    ] + arg_list
    
    run_image_instance_segmentation_parser = parser.add_parser(
        "image-instance-segmentation", help="✨ Run AutoTrain Image Instance Segmentation"
    )
    for arg in arg_list:
        names = [arg["arg"]] + arg.get("alias", [])
        if "action" in arg:
            run_image_instance_segmentation_parser.add_argument(
                *names,
                dest=arg["arg"].replace("--", "").replace("-", "_"),
                help=arg["help"],
                required=arg.get("required", False),
                action=arg.get("action"),
                default=arg.get("default"),
            )
        else:
            run_image_instance_segmentation_parser.add_argument(
                *names,
                dest=arg["arg"].replace("--", "").replace("-", "_"),
                help=arg["help"],
                required=arg.get("required", False),
                type=arg.get("type"),
                default=arg.get("default"),
            )
    
    run_image_instance_segmentation_parser.set_defaults(func=run_image_instance_segmentation_command_factory) 