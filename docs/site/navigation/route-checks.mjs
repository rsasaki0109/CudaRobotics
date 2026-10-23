export function validateRoute(route, backend) {
  if (!route || !['ready', 'no_path', 'start_unsafe', 'goal_unsafe', 'endpoint_grid_clearance'].includes(route.status)
    || !Number.isFinite(route.length) || route.length < 0 || !Array.isArray(route.path)
    || !route.path.every(point => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite))
    || (route.status === 'ready' ? route.path.length < 2 || !Number.isFinite(route.minimumClearance)
      : route.path.length !== 0 || route.length !== 0 || route.minimumClearance !== null)) {
    throw new Error(`Invalid ${backend.toUpperCase()} route`);
  }
}

export function routeAgreement(firstRoutes, secondRoutes, tolerance = 1e-6) {
  if (!secondRoutes) return null;
  if (!firstRoutes.length || firstRoutes.length !== secondRoutes.length) throw new Error('Expected paired routes');
  const accuracy = {matches: true, checkedPairs: firstRoutes.length, mismatchedPairs: 0,
    statusMismatches: 0, pathMismatches: 0, clearanceMismatches: 0, maxWaypointErrorMetres: 0, maxLengthErrorMetres: 0,
    maxClearanceErrorMetres: 0, toleranceMetres: tolerance};
  for (let index = 0; index < firstRoutes.length; index++) {
    const first = firstRoutes[index];
    const second = secondRoutes[index];
    const statusMatches = first.status === second.status;
    let pathMatches = first.path.length === second.path.length;
    if (pathMatches) {
      for (let waypoint = 0; waypoint < first.path.length; waypoint++) {
        const error = Math.hypot(first.path[waypoint][0] - second.path[waypoint][0], first.path[waypoint][1] - second.path[waypoint][1]);
        accuracy.maxWaypointErrorMetres = Math.max(accuracy.maxWaypointErrorMetres, error);
        if (error > tolerance) pathMatches = false;
      }
    }
    const lengthError = Math.abs(first.length - second.length);
    const clearanceError = first.minimumClearance === null || second.minimumClearance === null
      ? first.minimumClearance === second.minimumClearance ? 0 : Infinity
      : Math.abs(first.minimumClearance - second.minimumClearance);
    accuracy.maxLengthErrorMetres = Math.max(accuracy.maxLengthErrorMetres, lengthError);
    if (Number.isFinite(clearanceError)) accuracy.maxClearanceErrorMetres = Math.max(accuracy.maxClearanceErrorMetres, clearanceError);
    if (!statusMatches) accuracy.statusMismatches++;
    if (!pathMatches) accuracy.pathMismatches++;
    if (clearanceError > tolerance) accuracy.clearanceMismatches++;
    if (!statusMatches || !pathMatches || lengthError > tolerance || clearanceError > tolerance) accuracy.mismatchedPairs++;
  }
  accuracy.matches = accuracy.mismatchedPairs === 0;
  return accuracy;
}
