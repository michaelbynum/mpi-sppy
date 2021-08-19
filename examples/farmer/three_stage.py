import pyomo.environ as pe
import mpisppy.utils.sputils as sputils
from mpisppy import scenario_tree
from mpisppy.opt import ef, sc
import logging
from mpi4py import MPI
import random
from rich.console import Console
from rich.table import Table


comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


"""
First stage - decide how much land to buy; total_acreage is first stage variable
multi-year problem
Each year is a second stage - decide how much land to devote to each crop; 
the prices vary from year to year, but known each year;
devoted_acreage is the second stage variable
"""


def tuple_to_scenario_name(year, crop_yield):
    return 'root_' + str(int(year)) + '_' + crop_yield


def scenario_name_to_tuple(scenario_name: str):
    s = scenario_name.split('_')
    year = int(s[1])
    crop_yield = s[2]
    return year, crop_yield


class Farmer(object):
    def __init__(self):
        self.crops = ['WHEAT', 'CORN', 'SUGAR_BEETS']
        self.min_acreage = 0
        self.max_acreage = 1000
        self.land_cost = 10000
        self.years = [1, 2, 3, 4, 5]
        self.PriceQuota = dict()
        self.SubQuotaSellingPrice = dict()
        self.SuperQuotaSellingPrice = dict()
        self.CattleFeedRequirement = dict()
        self.PurchasePrice = dict()
        self.PlantingCostPerAcre = dict()
        for year in self.years:
            random.seed(year)
            l, u = 500, 2000
            self.PriceQuota[year] = dict(WHEAT=random.uniform(l, u), CORN=random.uniform(l, u), SUGAR_BEETS=random.uniform(l, u))
            l, u = 100, 300
            self.SubQuotaSellingPrice[year] = dict(WHEAT=random.uniform(l, u), CORN=random.uniform(l, u), SUGAR_BEETS=random.uniform(l, u))
            l, u = 1, 50
            self.SuperQuotaSellingPrice[year] = dict(WHEAT=random.uniform(l, u), CORN=random.uniform(l, u), SUGAR_BEETS=random.uniform(l, u))
            l, u = 100, 500
            self.CattleFeedRequirement[year] = dict(WHEAT=random.uniform(l, u), CORN=random.uniform(l, u), SUGAR_BEETS=0)
            l, u = 300, 1000
            self.PurchasePrice[year] = dict(WHEAT=random.uniform(l, u), CORN=random.uniform(l, u), SUGAR_BEETS=random.uniform(l, u))
            l, u = 100, 300
            self.PlantingCostPerAcre[year] = dict(WHEAT=random.uniform(l, u), CORN=random.uniform(l, u), SUGAR_BEETS=random.uniform(l, u))

        self.crop_yield = dict()
        self.crop_yield['Low'] = {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0}
        self.crop_yield['Average'] = {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0}
        self.crop_yield['High'] = {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}

        self.scenarios = list()
        for year in self.years:
            for crop_yield in self.crop_yield.keys():
                self.scenarios.append(tuple_to_scenario_name(year, crop_yield))


def create_scenario(scenario: str, farmer: Farmer):
    m = pe.ConcreteModel()

    m.total_acreage = pe.Var(bounds=(farmer.min_acreage, farmer.max_acreage))
    m.crops = pe.Set(initialize=farmer.crops)
    m.devoted_acreage = pe.Var(m.crops, bounds=(0, farmer.max_acreage))
    m.total_acreage_con = pe.Constraint(expr=sum(m.devoted_acreage.values()) <= m.total_acreage)

    m.QuantitySubQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
    m.QuantitySuperQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
    m.QuantityPurchased = pe.Var(m.crops, bounds=(0.0, None))

    year, crop_yield = scenario_name_to_tuple(scenario)

    def EnforceCattleFeedRequirement_rule(m, i):
        return (farmer.CattleFeedRequirement[year][i] <= (farmer.crop_yield[crop_yield][i] * m.devoted_acreage[i]) +
                m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i])
    m.EnforceCattleFeedRequirement = pe.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(m, i):
        return (m.QuantitySubQuotaSold[i] +
                m.QuantitySuperQuotaSold[i] -
                (farmer.crop_yield[crop_yield][i] * m.devoted_acreage[i]) <= 0.0)
    m.LimitAmountSold = pe.Constraint(m.crops, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(m, i):
        return 0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[year][i]
    m.EnforceQuotas = pe.Constraint(m.crops, rule=EnforceQuotas_rule)

    obj_scale = 0.001
    # 20 years
    third_stage_obj_expr = obj_scale * 20 * sum(farmer.PurchasePrice[year][crop] * m.QuantityPurchased[crop] for crop in m.crops)
    third_stage_obj_expr -= obj_scale * 20 * sum(farmer.SubQuotaSellingPrice[year][crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops)
    third_stage_obj_expr -= obj_scale * 20 * sum(farmer.SuperQuotaSellingPrice[year][crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops)
    second_stage_obj_expr = obj_scale * 20 * sum(farmer.PlantingCostPerAcre[year][crop] * m.devoted_acreage[crop] for crop in m.crops)
    first_stage_obj_expr = obj_scale * farmer.land_cost * m.total_acreage

    m.first_stage_obj_expr = pe.Expression(expr=first_stage_obj_expr)
    m.second_stage_obj_expr = pe.Expression(expr=second_stage_obj_expr)
    m.third_stage_obj_expr = pe.Expression(expr=third_stage_obj_expr)

    m.obj = pe.Objective(expr=m.first_stage_obj_expr + m.second_stage_obj_expr + m.third_stage_obj_expr)

    # sputils.attach_root_node(m, m.first_stage_obj_expr, [m.devoted_acreage])
    m._mpisppy_node_list = [scenario_tree.ScenarioNode(name='ROOT',
                                                       cond_prob=1,
                                                       stage=1,
                                                       cost_expression=m.first_stage_obj_expr,
                                                       scen_name_list=farmer.scenarios,
                                                       nonant_list=[m.total_acreage],
                                                       scen_model=m,
                                                       parent_name=None),
                            scenario_tree.ScenarioNode(name='ROOT_' + str(year-1),
                                                       cond_prob=1/len(farmer.years),
                                                       stage=2,
                                                       cost_expression=m.second_stage_obj_expr,
                                                       scen_name_list=[i for i in farmer.scenarios if scenario_name_to_tuple(i)[0] == year],
                                                       nonant_list=[m.devoted_acreage],
                                                       scen_model=m,
                                                       parent_name='ROOT')
                            ]

    return m


def solve_individual_scenarios():
    farmer = Farmer()
    console = Console()
    table = Table(width=12*12)
    table.add_column('Year', width=12)
    table.add_column('Yield', width=12)
    table.add_column('Total Acreage', width=12)
    table.add_column('Wheat Acreage', width=12)
    table.add_column('Corn Acreage', width=12)
    table.add_column('SB Acreage', width=12)
    table.add_column('Objective', width=12)
    table.add_column('Obj Stage 1', width=12)
    table.add_column('Obj Stage 2', width=12)
    table.add_column('Obj Stage 3', width=12)
    for year in farmer.years:
        for crop_yield in farmer.crop_yield.keys():
            scenario_name = tuple_to_scenario_name(year, crop_yield)
            m = create_scenario(scenario_name, farmer)
            opt = pe.SolverFactory('gurobi_direct')
            res = opt.solve(m)
            pe.assert_optimal_termination(res)
            table.add_row(str(year),
                          crop_yield,
                          f"{m.total_acreage.value:.2e}",
                          f"{m.devoted_acreage['WHEAT'].value:.2e}",
                          f"{m.devoted_acreage['CORN'].value:.2e}",
                          f"{m.devoted_acreage['SUGAR_BEETS'].value:.2e}",
                          f"{pe.value(m.obj):.3e}",
                          f"{pe.value(m.first_stage_obj_expr):.3e}",
                          f"{pe.value(m.second_stage_obj_expr):.3e}",
                          f"{pe.value(m.third_stage_obj_expr):.3e}",
                          )
    console.print(table)


def solve_with_extensive_form():
    farmer = Farmer()
    options = dict()
    options['solver'] = 'gurobi_direct'
    options['branching_factors'] = [5, 3]
    scenario_kwargs = dict()
    scenario_kwargs['farmer'] = farmer

    ef_m = sputils.create_EF(farmer.scenarios,
                           create_scenario,
                           scenario_kwargs)
    opt = pe.SolverFactory('gurobi_direct')
    res = opt.solve(ef_m, tee=False)
    pe.assert_optimal_termination(res)
    console = Console()
    table = Table(width=12*12)
    table.add_column('Year', width=12)
    table.add_column('Yield', width=12)
    table.add_column('Total Acreage', width=12)
    table.add_column('Wheat Acreage', width=12)
    table.add_column('Corn Acreage', width=12)
    table.add_column('SB Acreage', width=12)
    for b in ef_m.block_data_objects(active=True, descend_into=True):
        if not b.name.startswith('root'):
            continue
        year, crop_yield = scenario_name_to_tuple(b.name)
        table.add_row(str(year),
                      crop_yield,
                      f"{b.total_acreage.value:.2e}",
                      f"{b.devoted_acreage['WHEAT'].value:.2e}",
                      f"{b.devoted_acreage['CORN'].value:.2e}",
                      f"{b.devoted_acreage['SUGAR_BEETS'].value:.2e}"
                      )
    console.print(table)

    opt = ef.ExtensiveForm(options=options,
                           all_scenario_names=farmer.scenarios,
                           scenario_creator=create_scenario,
                           scenario_creator_kwargs=scenario_kwargs,
                           all_nodenames=['ROOT'] + ['ROOT_' + str(i-1) for i in farmer.years])
    results = opt.solve_extensive_form()
    opt.report_var_values_at_rank0()


def solve_with_SC():
    farmer = Farmer()
    options = dict()
    options['branching_factors'] = [5, 3]
    scenario_kwargs = dict()
    scenario_kwargs['farmer'] = farmer

    opt = sc.SchurComplement(options=options,
                             all_scenario_names=farmer.scenarios,
                             scenario_creator=create_scenario,
                             scenario_creator_kwargs=scenario_kwargs,
                             all_nodenames=['ROOT'] + ['ROOT_' + str(i-1) for i in farmer.years])
    results = opt.solve()
    opt.report_var_values_at_rank0()


if __name__ == '__main__':
    # solve_with_extensive_form()
    # solve_individual_scenarios()
    solve_with_SC()
