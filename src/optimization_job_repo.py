from pymongo import MongoClient

from data import StockOptimizationJob, OptimizationResult, OptimizationJobStatus


class OptimizationRepository:

    def __init__(self):
        client = MongoClient("mongodb://localhost:27017")
        self.db = client['optimization']
        self.stock_prices = self.db["stock_price_collection"]
        self.jobs = self.db["jobs_collection"]
        self.results = self.db["results_collection"]

    def get_job(self, id: str):
        res = self.jobs.find_one({"_id": id})
        try:
            return StockOptimizationJob.from_dict(res)
        except:
            return None

    def save_job(self, job: StockOptimizationJob):
        self.jobs.update_one({"_id": job._id}, {"$set": job.to_dict()}, upsert=True)

    def get_saved_stock_price(self, date: str, stock_name: str):
        key = date + "__" + stock_name
        res = self.stock_prices.find_one({"_id": key})
        try:
            return res["price"]
        except:
            return None

    def save_stock_price(self, date: str, stock_name: str, price: float):
        key = date + "__" + stock_name
        self.stock_prices.update_one({"_id": key}, {"$set": {"price": price}}, upsert=True)


    def get_opt_result(self, id: str):
        res = self.results.find_one({"_id": id})
        try:
            return OptimizationResult.from_dict(res)
        except:
            return None

    def save_opt_result(self, result: OptimizationResult):
        self.stock_prices.update_one({"_id": result._id}, {"$set": result.to_dict()}, upsert=True)


if __name__ == '__main__':
    optRepo = OptimizationRepository()
    testJob = StockOptimizationJob(["test1", "test2"], [1, 3], 100, 30, _id="test1")
    optRepo.save_job(testJob)
    print(optRepo.get_job(testJob._id))

    # optRepo.save_stock_price("05-03-2022", "APL", 12.22)
    # print(optRepo.get_saved_stock_price("05-03-2022", "APL"))