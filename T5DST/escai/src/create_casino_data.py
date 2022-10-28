from pprint import pprint
import wandb

from convokit import Corpus, download

from config import get_args


def create_casino_data(args):
    corpus = Corpus(filename=download("casino-corpus", data_dir="~/myblue/.casino"))
    pprint(corpus.random_utterance())


if __name__ == "__main__":
    args = get_args()
    config = vars(args)
    config["dataset"] = "casino"
    wandb.init(
        id=args.wandb_run_id,
        project="collaborative-dst",
        entity="msamogh",
        config=vars(args),
    )

    create_casino_data(args)
