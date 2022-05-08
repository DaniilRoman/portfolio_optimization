import threading
from datetime import datetime, timedelta

round_precise = 5


def print_step_stats(pop):
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)


def get_current_date_str():
    return str(get_current_date())


def get_current_date():
    # return datetime.date(datetime.now()) - timedelta(days=40)
    return datetime.strptime("2021-06-01", "%Y-%m-%d").date()


# 11 - 2022-04-27

# 37 - 2022-04-01
# 40 - 2022-03-29
# 2021-06-01


def get_next_day(days, start_day: str = None):
    if start_day is None:
        date = get_current_date()
    else:
        date = datetime.strptime(start_day, "%Y-%m-%d").date()
    return str(date + timedelta(days=days))


def get_prev_day(days, start_day: str = None):
    if start_day is None:
        date = get_current_date()
    else:
        date = datetime.strptime(start_day, "%Y-%m-%d").date()
    return str(date - timedelta(days=days))


class OnlyPutBlockingQueue(object):

    def __init__(self):
        self.queue = {}
        self.cv = threading.Condition()

    def put(self, key, value):
        with self.cv:
            self.queue[key] = value
            self.cv.notify()
