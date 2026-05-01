import asyncio
import os
from argparse import ArgumentParser

from config import get_config
from treesearch.search import TreeSearch
from utils.log import _ROOT_LOGGER, attach_file_handler, set_log_level
from utils.path import mkdir
from utils.checks import require_executable
from treesearch.utils.costs_tracker import get_cost_tracker
import asyncio
from utils.statistics_tracker import get_statistics_tracker

logger = _ROOT_LOGGER.getChild("main")
cost_tracker = get_cost_tracker()
statistics_tracker = get_statistics_tracker()


async def main():
    set_log_level(os.getenv("ISGSA_LOG", "INFO"))

    config = get_config()
    out_dir = mkdir(config.out_dir)
    args = get_args()
    if args.init:
        mkdir(out_dir / "workspace")
        return

    attach_file_handler(out_dir)

    cost_tracker.set_out_dir(out_dir)
    statistics_tracker.set_out_dir(out_dir)
    
    require_executable("dot")

    user_req_lines: list[str] = []
    print('Enter you request, write "!start" to start:')
    while True:
        line = input("> ")
        if line.lower().strip().startswith("!start"):
            break
        user_req_lines.append(line)

    user_request = "\n".join(user_req_lines)
    
    logger.info("Starting AutoRecLab...")

    ts = TreeSearch(user_request, config=config)
    await ts._async_init()
    await ts.run()

    cost_tracker.saveSummarized()
    statistics_tracker.summarize_statistics()


def get_args():
    parser = ArgumentParser("AutoRecLab")
    parser.add_argument("--init", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
