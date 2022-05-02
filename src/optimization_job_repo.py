from typing import Optional

from pymongo import MongoClient

from data import StockOptimizationJob, OptimizationResult


class OptimizationRepository:

    def __init__(self):
        client = MongoClient("mongodb://localhost:27017")
        self.db = client['optimization']
        self.stock_prices = self.db["stock_prices"]
        self.stock_predicted_prices = self.db["stock_predicted_prices"]
        self.jobs = self.db["jobs"]
        self.results = self.db["results"]

    def get_job(self, id: str):
        res = self.jobs.find_one({"_id": id})
        try:
            return StockOptimizationJob.from_dict(res)
        except:
            return None

    def save_job(self, job: StockOptimizationJob):
        self.jobs.update_one({"_id": job._id}, {"$set": job.to_dict()}, upsert=True)

    def get_saved_stock_price(self, date: str, stock_name: str) -> Optional[float]:
        key = date + "__" + stock_name
        res = self.stock_prices.find_one({"_id": key})
        try:
            return res["price"]
        except:
            return None

    def save_stock_price(self, date: str, stock_name: str, price: float):
        key = date + "__" + stock_name
        self.stock_prices.update_one({"_id": key}, {"$set": {"price": price}}, upsert=True)

    def get_saved_predicted_stock_price(self, date: str, stock_name: str) -> Optional[float]:
        key = date + "__" + stock_name
        res = self.stock_predicted_prices.find_one({"_id": key})
        try:
            return res["price"]
        except:
            return None

    def save_stock_predicted_price(self, date: str, stock_name: str, price: float):
        key = date + "__" + stock_name
        self.stock_predicted_prices.update_one({"_id": key}, {"$set": {"price": price}}, upsert=True)

    def get_opt_result(self, id: str):
        res = self.results.find_one({"_id": id})
        try:
            return OptimizationResult.from_dict(res)
        except:
            return None

    def save_opt_result(self, result: OptimizationResult):
        self.results.update_one({"_id": result._id}, {"$set": result.to_dict()}, upsert=True)
