from mpisppy.spbase import SPBase
import pyomo.environ as pyo
import logging
import parapint
from typing import List, Callable, Dict, Optional, Tuple, Any, Union
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.visitor import identify_variables
from mpi4py import MPI
from mpisppy.utils.sputils import find_active_objective
from pyomo.common.collections.component_set import ComponentSet


logger = logging.getLogger('mpisppy.sc')


SCOptions = parapint.algorithms.IPOptions


def _assert_continuous(m: _BlockData):
    for v in m.component_data_objects(pyo.Var, descend_into=True, active=True):
        if not v.is_continuous():
            raise RuntimeError(f'Variable {v} in block {m} is not continuous; The Schur-Complement method only supports continuous problems.')


def _get_all_used_unfixed_vars(m: _BlockData):
    res = ComponentSet()
    for c in m.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        res.update(v for v in identify_variables(c.body, include_fixed=False))
    res.update(identify_variables(find_active_objective(m).expr, include_fixed=False))
    return res


class _SCInterface(parapint.interfaces.MPIStochasticSchurComplementInteriorPointInterface):
    def __init__(self,
                 local_scenario_models: Dict[str, _BlockData],
                 all_scenario_names: List[str],
                 comm: MPI.Comm,
                 ownership_map: Dict):
        self.local_scenario_models = local_scenario_models

        self._used_unfixed_vars_by_scenario = dict()

        nonant_var_identifiers = list()
        nonant_var_identifiers_set = set()
        for scen_name, scen_model in self.local_scenario_models.items():
            scen_variables = _get_all_used_unfixed_vars(scen_model)
            self._used_unfixed_vars_by_scenario[scen_name] = scen_variables
            for nonant_id, nonant_var in scen_model._mpisppy_data.nonant_indices.items():
                if nonant_var in scen_variables:
                    if nonant_id not in nonant_var_identifiers_set:
                        nonant_var_identifiers.append(nonant_id)
                        nonant_var_identifiers_set.add(nonant_id)

        tmp = comm.allgather(nonant_var_identifiers)
        tmp2 = list()
        for i in tmp:
            tmp2.extend(i)
        tmp = tmp2
        del tmp2
        nonant_var_identifiers = list()
        nonant_var_identifiers_set = set()
        for nonant_id in tmp:
            if nonant_id not in nonant_var_identifiers_set:
                nonant_var_identifiers.append(nonant_id)
                nonant_var_identifiers_set.add(nonant_id)

        super(_SCInterface, self).__init__(scenarios=all_scenario_names,
                                           nonanticipative_var_identifiers=nonant_var_identifiers,
                                           comm=comm,
                                           ownership_map=ownership_map)

    def build_model_for_scenario(self,
                                 scenario_identifier: str) -> Tuple[_BlockData, Dict[Any, _GeneralVarData]]:
        m = self.local_scenario_models[scenario_identifier]

        _assert_continuous(m)

        active_obj = find_active_objective(m)
        active_obj.deactivate()
        m._mpisppy_model.weighted_obj = pyo.Objective(expr=m._mpisppy_probability * active_obj.expr, sense=active_obj.sense)

        nonant_vars = dict()
        scen_variables = self._used_unfixed_vars_by_scenario[scenario_identifier]
        for nonant_id, _nonant_var in m._mpisppy_data.nonant_indices.items():
            if _nonant_var in scen_variables:
                nonant_vars[nonant_id] = _nonant_var

        return m, nonant_vars


class SchurComplement(SPBase):
    def __init__(self,
                 options: Union[Dict, SCOptions],
                 all_scenario_names: List,
                 scenario_creator: Callable,
                 scenario_creator_kwargs: Optional[Dict] = None,
                 all_nodenames=None,
                 mpicomm=None,
                 model_name=None,
                 suppress_warnings=False):
        super(SchurComplement, self).__init__(options=options,
                                              all_scenario_names=all_scenario_names,
                                              scenario_creator=scenario_creator,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              all_nodenames=all_nodenames,
                                              mpicomm=mpicomm)

        if self.bundling:
            raise ValueError('The Schur-Complement method does not support bundling')

        ownership_map = dict()
        for _rank, scenario_index_list in enumerate(self._rank_slices):
            for _scenario_ndx in scenario_index_list:
                ownership_map[_scenario_ndx] = _rank

        self.interface = _SCInterface(local_scenario_models=self.local_scenarios,
                                      all_scenario_names=self.all_scenario_names,
                                      comm=self.mpicomm,
                                      ownership_map=ownership_map)

    def solve(self):
        if isinstance(self.options, SCOptions):
            options = self.options()
        else:
            options = dict(self.options)
            options.pop('branching_factors', None)
            options = SCOptions()(options)
        if options.linalg.solver is None:
            options.linalg.solver = parapint.linalg.MPISchurComplementLinearSolver(
                subproblem_solvers={ndx: parapint.linalg.InteriorPointMA27Interface(cntl_options={1: 1e-6}) for ndx in range(len(self.all_scenario_names))},
                schur_complement_solver=parapint.linalg.InteriorPointMA27Interface(cntl_options={1: 1e-6}))

        status = parapint.algorithms.ip_solve(interface=self.interface,
                                              options=options)
        if status != parapint.algorithms.InteriorPointStatus.optimal:
            raise RuntimeError('Schur-Complement Interior Point algorithm did not converge')

        self.interface.load_primals_into_pyomo_model()

        return status
