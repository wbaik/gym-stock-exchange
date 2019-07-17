import git
import os


def get_git_root():
    return git.Repo(os.getcwd(), search_parent_directories=True)\
              .git\
              .rev_parse("--show-toplevel")


def get_tickers():
    ticker_list = yaml.load(open(YAML_FILE_PATH, 'r'), Loader=yaml.Loader)['ticker']
    assert ticker_list, "Please add tickers into {}".format(YAML_FILE_PATH)

    return ticker_list


YAML_FILE_NAME = 'temp.yaml'
YAML_FILE_PATH = get_git_root() + '/' + YAML_FILE_NAME
