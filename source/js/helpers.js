// Coordinate Transformations 
function canvasToDisk(coord) {
    x = Math.round(coord.x - main_x) / main_radius
    y = -Math.round(coord.y - main_y) / main_radius

    if (x * x + y * y < 1.0) {
        return {x: x, y: y}
    } else {
        return null
    }
}

function diskToCanvas(coord) {
    x = coord.x * main_x + main_radius
    y = -coord.y * main_x + main_radius
    return {x: x, y: y}
}

function polarToCartesian(centerX, centerY, radius, angleInDegrees) {
  var angleInRadians = angleInDegrees;

  return {
    x: centerX + (radius * Math.cos(angleInRadians)),
    y: centerY + (radius * Math.sin(angleInRadians))
  };
}

function describeArc(x, y, radius, startAngle, endAngle) {
    minAngle = Math.min(startAngle, endAngle);
    maxAngle = Math.max(startAngle, endAngle);
    var start = polarToCartesian(x, y, radius, minAngle);
    var end = polarToCartesian(x, y, radius, maxAngle);
    var largeArcFlag = "0";

    // Handle angle wrapping around 360
    if ((x < main_x && y < main_y) || (x < main_x && y >= main_y)) {
        sweepFlag = (minAngle < Math.PI) && (maxAngle > Math.PI) ? "0": "1";
    } else {
        sweepFlag = "1";
    }

    var d = [
        "M", start.x, start.y, 
        "A", radius, radius, 0, largeArcFlag, sweepFlag, end.x, end.y
    ].join(" ");

    return d;       
}

// Hyperbolic lines
function computeHyperbolicLine(p, q, center, radius) {
    // Infinite case
    if (p.x == 0 && p.y == 0 && q.x == 0 && q.y == 0) {
        return null
    }

    function lineEqn(p, q) {
        line = {a: (p.y - q.y), b: (q.x - p.x), c: (p.x * q.y - q.x * p.y)}
        if (Math.abs(line.b) > 0.001) {
            return {a: line.a / line.b, b: line.b / line.b, c: line.c / line.b}
        } else {
            return line
        }
    }

    function perpLine(pq, P) {
        return {a: pq.b, b: -pq.a, c: -P.x * pq.b + P.y * pq.a}
    }

    function dist(p, q) {
        return Math.sqrt(Math.pow(p.x - q.x, 2) + Math.pow(p.y - q.y, 2))
    }

    function midpoint(p, q) {
        return {x: (p.x + q.x) / 2, y: (p.y + q.y) / 2}
    }

    function inversion(p, center, radius) {
        pDist = dist(p, {x: center.x, y: center.y})
        c =  radius * radius / (pDist * pDist)
        uVector = {x: (p.x - center.x), y: (p.y - center.y)}
        pp = {x: c * uVector.x + center.x, y: c * uVector.y + center.y}
        return pp
    }

    function intersectLines(l1, l2) {
        a = l1.a
        b = l1.b
        c = l1.c
        j = l2.a
        k = l2.b
        l = l2.c
        return {
            x: (c * k - b * l) / (b * j - a * k),
            y: (a * l - c * j) / (b * j - a * k)
        }
    }

    // Algorithm from:
    // https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model#Compass_and_straightedge_construction
    P = p
    Q = q
    Pprime = inversion(P, {x: center.x, y: center.y}, radius)
    Qprime = inversion(Q, {x: center.y, y: center.y}, radius)
    M = midpoint(P, Pprime)
    N = midpoint(Q, Qprime)
    m = perpLine(lineEqn(P, Pprime), M)
    n = perpLine(lineEqn(Q, Qprime), N)
    C = intersectLines(m, n)

  
    function atan2(y, x) {
        angle = Math.atan2(y, x)
        return (angle >= 0 ? angle : (2*Math.PI + angle))
    }

    endAngle = atan2((P.y - C.y), P.x - C.x)
    startAngle = atan2((Q.y - C.y), Q.x - C.x)

    return {
        center: C,
        startAngle: startAngle, 
        endAngle: endAngle,
        radius: dist(C, P)
    }
}

function hDist(p, q) {
    // From: https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model#Distance
    var numer = 2 * (Math.pow(p.x - q.x, 2) + Math.pow(p.y - q.y, 2));
    var denom = ((1 - (Math.pow(p.x, 2) + Math.pow(p.y, 2)))
                 * (1 - (Math.pow(q.x, 2) + Math.pow(q.y, 2))));

    return Math.acosh(1 + numer / denom)
}
