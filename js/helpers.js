// Coordinate Transformations 
function canvasToDisk(coord) {
    x = (coord.x - main_x) / main_radius
    y = -(coord.y - main_y) / main_radius

    if (x * x + y * y < 1.0) {
        return {x: x, y: y}
    } else {
        return null
    }
}

function diskToCanvas(coord) {
    x = coord.x * main_radius + main_x
    y = -coord.y * main_radius + main_y
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

function midpoint(p, q) {
    return {x: (p.x + q.x) / 2, y: (p.y + q.y) / 2}
}

function eDist(p, q) {
    return Math.sqrt(Math.pow(p.x - q.x, 2) + Math.pow(p.y - q.y, 2))
}

// Hyperbolic lines
function computeHyperbolicLine(id, p, q, center, radius) {
    // Infinite case
    if (p.x == 0 && p.y == 0 && q.x == 0 && q.y == 0) {
        return null
    }

    function inversion(p, center, radius) {
        pDist = eDist(p, {x: center.x, y: center.y})
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
        id: id,
        drawType: "line",
        p: p,
        q: q,
        center: C,
        startAngle: startAngle, 
        endAngle: endAngle,
        radius: eDist(C, P)
    }
}

function hDist(p, q) {
    // From: https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model#Distance
    var numer = 2 * (Math.pow(p.x - q.x, 2) + Math.pow(p.y - q.y, 2));
    var denom = ((1 - (Math.pow(p.x, 2) + Math.pow(p.y, 2)))
                 * (1 - (Math.pow(q.x, 2) + Math.pow(q.y, 2))));

    return Math.acosh(1 + numer / denom)
}

function computeHyperbolicCircle(id, p, q) {
    function findPoint(startPoint, endPoint, l, radius) {
        objective = function(z) {
            otherPoint = {x: z[0], y: -l.a / l.b * z[0] - l.c / l.b}
            return Math.pow(hDist(startPoint, otherPoint) - radius, 2)
        }

        tol = 1e10; 
        resultPoint = startPoint;
        i = 1;
        
        dist = (endPoint != null) ? hDist(startPoint, endPoint) : -1;
        ePoint = (endPoint != null) ? endPoint : startPoint;
        while ((tol > 0.001 && i < 100) || 
               (tol < 0.001 && hDist(ePoint, resultPoint) < dist)) {

            if (endPoint != null)
                x0 = [startPoint.x - 0.005 * i * (ePoint.x - startPoint.x)];
            else
                x0 = [startPoint.x + 0.005 * i * startPoint.x];

            sol = numeric.uncmin(objective, x0);
            resultPoint = {
                x: sol.solution[0], 
                y: -l.a / l.b * sol.solution[0] - l.c / l.b
            }
            tol = sol.f;
            i = i + 1;
        }
        return resultPoint;
    }

    function findPoint2(startPoint, l, radius) {
        objective = function(z) {
            otherPoint = {x: z[0], y: -l.a / l.b * z[0] - l.c / l.b}
            return Math.pow(hDist(startPoint, otherPoint) - radius, 2)
        }

        tol = 1e10; 
        resultPoint = startPoint;
        i = 1;
        
        while (tol > 0.001 && i < 1000) {
            x0 = [startPoint.x + Math.random() / 100];
            sol = numeric.uncmin(objective, x0);
            resultPoint = {
                x: sol.solution[0], 
                y: -l.a / l.b * sol.solution[0] - l.c / l.b
            }
            tol = sol.f;
            i = i + 1;
        }
        return resultPoint;
    }

    function findCircle(p1, p2, p3) {
        // Algorithm from: http://www.ambrsoft.com/trigocalc/circle3d.htm 
        A = p1.x * (p2.y - p3.y) - p1.y * (p2.x - p3.x) + p2.x * p3.y - p3.x * p2.y;
        B = ((Math.pow(p1.x, 2) + Math.pow(p1.y, 2)) * (p3.y - p2.y) +
             (Math.pow(p2.x, 2) + Math.pow(p2.y, 2)) * (p1.y - p3.y) +
             (Math.pow(p3.x, 2) + Math.pow(p3.y, 2)) * (p2.y - p1.y))
        C = ((Math.pow(p1.x, 2) + Math.pow(p1.y, 2)) * (p2.x - p3.x) +
             (Math.pow(p2.x, 2) + Math.pow(p2.y, 2)) * (p3.x - p1.x) +
             (Math.pow(p3.x, 2) + Math.pow(p3.y, 2)) * (p1.x - p2.x))
        D = ((Math.pow(p1.x, 2) + Math.pow(p1.y, 2)) * (p3.x * p2.y - p2.x * p3.y) +
             (Math.pow(p2.x, 2) + Math.pow(p2.y, 2)) * (p1.x * p3.y - p3.x * p1.y) +
             (Math.pow(p3.x, 2) + Math.pow(p3.y, 2)) * (p2.x * p1.y - p1.x * p2.y))

        return {
            cx: -B / (2 * A),
            cy: -C / (2 * A),
            radius: Math.sqrt((B*B + C*C - 4 * A * D) / (4 * A * A))
        }
    }

    // All calculations in terms of pointcare disk
    startPoint = canvasToDisk(p);
    p1 = canvasToDisk(q);

    radius = hDist(startPoint, p1);
    line = lineEqn(startPoint, p1);
    p2 = findPoint(startPoint, p1, line, radius);

    pline = perpLine(line, startPoint);
    p3 = findPoint(startPoint, null, pline, radius);

    circle = findCircle(p1, p2, p3);
    center = diskToCanvas({x: circle.cx, y: circle.cy});

    return {
        id,
        drawType: "circle",
        p: p,
        q: q,
        start: diskToCanvas(startPoint), 
        p1: diskToCanvas(p1), 
        p2: diskToCanvas(p2),
        p3: diskToCanvas(p3),
        center: center,
        radius: eDist(center, diskToCanvas(p1))
    }
}
