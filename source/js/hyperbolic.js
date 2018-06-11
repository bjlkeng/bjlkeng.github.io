// Globals
var w = window.innerWidth,
    h = window.innerHeight;

var main_x = 250,
    main_y = 250,
    main_radius = 200

var unitCircleAttrib = {
    cx: main_x,
    cy: main_y,
    r: main_radius,
    stroke: "black",
    strokewidth: "2",
    fill: "white",
    opacity: "1.0"
}

var elements = d3.select(".elements");
var allObjects = [];
var isMouseDown = false;
var tmpObject = null;
var srcCoord = null;

// Create main window and poincare disk
var svg = d3.select(".mainbox").append("svg").attrs({
    width: w,
    height: h
})
var unitCircle = svg.append("circle")
                    .attrs(unitCircleAttrib)

// Make gridlines
var xScale = d3.scaleLinear()
               .domain([-1, 1])
               .range([main_x - main_radius, main_x + main_radius]);
var yScale = d3.scaleLinear()
               .domain([1, -1])
               .range([main_y - main_radius, main_y + main_radius]);

function make_x_gridlines() {
    return d3.axisBottom(xScale)
             .ticks(10)
}

function make_y_gridlines() {
    return d3.axisLeft(xScale)
             .ticks(10)
}

svg.append("g")
   .attr("class", "grid")
   .attr("opacity", "0.2")
   .attr("transform", "translate(0," + (main_y + main_radius) + ")")
   .call(make_x_gridlines()
         .tickSize(-2 * main_radius)
         .tickFormat(""));

svg.append("g")
   .attr("class", "grid")
   .attr("opacity", "0.2")
   .attr("transform", "translate(" + (main_x - main_radius) + ", 0)")
   .call(make_y_gridlines()
         .tickSize(-2 * main_radius)
         .tickFormat(""));

svg.append("g")
   .attr("transform", "translate(0," + (main_y + main_radius) + ")")
   .call(d3.axisBottom(xScale));

svg.append("g")
   .attr("transform", "translate(" + (main_x - main_radius) + ", 0)")
   .call(make_y_gridlines())
   .call(d3.axisLeft(yScale));


// Handle line/circle buttons
var drawType = "line";
d3.select(".line-button")
  .classed("active", true)
  .on("click", function(d, i) { 
      drawType = "line"
      d3.select(".line-button")
        .classed("active", true)
      d3.select(".circle-button")
        .classed("active", false)
  });

d3.select(".circle-button")
  .classed("active", false)
  .on("click", function(d, i) { 
      drawType = "circle"
      d3.select(".circle-button")
        .classed("active", true);
      d3.select(".line-button")
        .classed("active", false);
  });

d3.select(".reset-button")
  .classed("active", false)
  .on("mousedown", function(d, i) { 
      allObjects = [];
      isMouseDown = false;
      tmpObject = null;
      drawElementsBox();
      drawObjects();
  })
  .on("mouseup", function(d) {
      d3.select(this)
        .classed("active", false);
      
  });

function getTmpObject(p, q) {
    label = drawType + "-" + (allObjects.filter(x => x.drawType == drawType).length + 1).toString();
    obj = null;
    if (drawType == "line")
        obj = computeHyperbolicLine(label, p, q, {x: main_x, y: main_y}, main_radius);
    else if (drawType == "circle")
        obj = computeHyperbolicCircle(label, p, q);

    obj.isTmp = true;
    return obj;
}

function highlightObject(obj) {
    tmpData = [];
    if (obj){
        tmpData = [obj];
        d3.select(".elementitem" + obj.id)
          .classed("active", true);
        d3.select("." + obj.id)
          .style("stroke", "red");
    } else {
        d3.selectAll(".elementitem")
          .classed("active", false);
        d3.selectAll(".objects")
          .style("stroke", "black");
    }

    g.selectAll(".line-tmp")
     .remove()
    g.selectAll(".circle-start")
     .remove()
    g.selectAll(".circle-tmp")
     .remove()

    if (obj && obj.drawType == "line") {
        g.selectAll(".line-tmp")
         .data(tmpData)
         .enter()
         .append("circle")
         .attr("class", "line-tmp")
         .attr("cx", function(hline) { 
             return hline.center.x; 
         })
         .attr("cy", function(hline) {
             return hline.center.y; 
         })
         .style("stroke", "black")
         .style("stroke-width", "2")
         .style("opacity", "0.2")
         .style("fill", "none")
         .attr("r", function(hline) {
            return hline.radius;
         })
    } else if (obj && obj.drawType == "circle") {
        g.selectAll(".circle-start")
         .data(tmpData)
         .enter()
         .append("circle")
         .attr("class", function(d) { return "circle-start circle-tmp"})
         .attr("cx", function(d) { return d.start.x; })
         .attr("cy", function(d) { return d.start.y; })
         .style("fill", "black")
         .attr("r", 3);
        
        g.selectAll(".circle-p1")
         .data(tmpData)
         .enter()
         .append("circle")
         .attr("class", function(d) { return "circle-p1 circle-tmp"})
         .attr("cx", function(d) { return d.p1.x; })
         .attr("cy", function(d) { return d.p1.y; })
         .style("fill", "black")
         .style("opacity", "0.5")
         .attr("r", 3);
        
        g.selectAll(".circle-p2")
         .data(tmpData)
         .enter()
         .append("circle")
         .attr("class", function(d) { return "circle-p2 circle-tmp"})
         .attr("cx", function(d) { return d.p2.x; })
         .attr("cy", function(d) { return d.p2.y; })
         .style("fill", "black")
         .style("opacity", "0.5")
         .attr("r", 3);
        
        g.selectAll(".circle-p3")
         .data(tmpData)
         .enter()
         .append("circle")
         .attr("class", function(d) { return "circle-p3 circle-tmp"})
         .attr("cx", function(d) { return d.p3.x; })
         .attr("cy", function(d) { return d.p3.y; })
         .style("fill", "black")
         .style("opacity", "0.5")
         .attr("r", 3);
        
        g.selectAll(".circle-center")
         .data(tmpData)
         .enter()
         .append("circle")
         .attr("class", function(d) { return "circle-center circle-tmp"})
         .attr("cx", function(d) { return d.center.x; })
         .attr("cy", function(d) { return d.center.y; })
         .style("fill", "orange")
         .style("opacity", "1.0")
         .attr("r", 3);
    }
}

function drawLines(data) {
    g.selectAll(".line-start")
        .remove();

    g.selectAll(".line-start")
     .data(data)
     .enter()
     .append("circle")
     .attr("class", function(d) { return "objects line-start "})
     .attr("cx", function(d) { return d.p.x; })
     .attr("cy", function(d) { return d.p.y; })
     .style("fill", "black")
     .attr("r", 3)
     .on("mouseover", function (d) {
         highlightObject(d);
     }).on("mousemove", function (d) {
         highlightObject(d);
     }).on("mouseleave", function (d) {
         highlightObject(null);
     })


    g.selectAll(".line-end")
     .remove()

    g.selectAll(".line-end")
     .data(data)
     .enter()
     .append("circle")
     .attr("class", function(d) { return "objects line-end "})
     .attr("cx", function(d) { return d.q.x; })
     .attr("cy", function(d) { return d.q.y; })
     .style("fill", "black")
     .attr("r", 3)
     .on("mouseover", function (d) {
         highlightObject(d);
     }).on("mousemove", function (d) {
         highlightObject(d);
     }).on("mouseleave", function (d) {
         highlightObject(null);
     })


    g.selectAll(".line-arc")
     .remove()

    g.selectAll(".line-arc")
     .data(data)
     .enter()
     .append("path")
     .attr("class", function(d) { return "objects line-arc " + d.id})
     .style("fill", "none")
     .style("stroke", function(d){ return d.isTmp ? "red" : "black";})
     .style("stroke-width", "2")
     .attr("d", function(d) { 
         return describeArc(d.center.x, d.center.y, d.radius,
                            d.startAngle, d.endAngle)
     }).on("mouseover", function (d) {
         highlightObject(d);
     }).on("mousemove", function (d) {
         highlightObject(d);
     }).on("mouseleave", function (d) {
         highlightObject(null);
     })
}

function drawCircles(data) {
    g.selectAll(".hCircle")
        .remove();
    g.selectAll(".hCircle")
        .data(data)
        .enter()
        .append("circle")
        .attr("class", function(d) {return "objects hCircle " + d.id})
        .attr("cx", function(d) { return d.center.x; })
        .attr("cy", function(d) { return d.center.y; })
        .style("stroke", function(d){ return d.isTmp ? "red" : "black";})
        .style("stroke-width", "2")
        .style("fill", "none")
        .attr("r", function(d) {
            return d.radius;
        }).on("mouseover", function (d) {
            highlightObject(d);
        }).on("mousemove", function (d) {
            highlightObject(d);
        }).on("mouseleave", function (d) {
            highlightObject(null);
        });

}

function drawObjects() {
    lineData = allObjects.filter(x => x.drawType == "line");
    circleData = allObjects.filter(x => x.drawType == "circle");
    if (drawType == "line") {
        if (tmpObject != null && tmpObject.p != null && tmpObject.q != null) {
            lineData = lineData.concat(tmpObject);
        }
    } else if (drawType == "circle") {
        if (tmpObject != null && tmpObject.p != null && tmpObject.q != null) {
            circleData = circleData.concat(tmpObject);
        }
    }
    highlightObject(tmpObject);
    drawLines(lineData);
    drawCircles(circleData)
}

function drawElementsBox() {
    sel = elements.selectAll(".elementitem")
                  .data(allObjects)

    sel.exit()
       .remove();
    
    t = sel.enter()
           .append("div")
           .attr("class", function (d) { return "elementitem list-group-item elementitem" + d.id})
           .on("mouseover", function (d) {
               highlightObject(d)
           })
           .on("mouseleave", function (d) {
               highlightObject(null)
           })


    t.append("div")
     .text(function(d) {return "ID: " + d.id })
     .append("br")
    t.append("div")
     .text(function(d) {
         if (d.drawType == "line") {
            p = canvasToDisk(d.p);
            return "Start: (" + p.x.toFixed(3) + ", " + p.y.toFixed(3) + ")";
         } else if (d.drawType == "circle") {
            p = canvasToDisk(d.start)
            return ("Center: (" + p.x.toFixed(3) + ", " + p.y.toFixed(3) + ")");
         }
     }).append("br")
    t.append("div")
     .text(function(d) {
         if (d.drawType == "line") {
            q = canvasToDisk(d.q);
            return "End: (" + q.x.toFixed(3) + ", " + q.y.toFixed(3) + ")";
         } else if (d.drawType == "circle") {
            return "Hyperbolic Radius: " + hDist(canvasToDisk(d.start), 
                                                 canvasToDisk(d.p1)).toFixed(3);
         }
     }).append("br")
    t.append("div")
        .text(function(d) {
            if (drawType == "line") {
                p = canvasToDisk(d.p);
                q = canvasToDisk(d.q);
                return "Hyperbolic Length:" + hDist(p, q).toFixed(3);
            }
        })
}

function drawInfoBox(canvasCoord) {
    diskCoord = canvasToDisk(canvasCoord);
    coordText1 = "";
    coordText2 = "";
    distText = "";

    if (diskCoord != null) {
        if (tmpObject != null) {
            p = canvasToDisk(tmpObject.p)
            q = canvasToDisk(tmpObject.q)
            coordText1 = "Start: " + "[" + p.x.toFixed(3) + ", " + p.y.toFixed(3) + "]";
            coordText2 = "End: " + "[" + q.x.toFixed(3) + ", " + q.y.toFixed(3) + "]";

            dist = hDist(p, q);
            type = drawType == "line" ? "Hyperbolic Distance: " : "Radius: "; 
            distText = type + dist.toFixed(3);
        } else if (!isMouseDown || srcCoord == null) {
            coordText1 = "Start: " + "[" + diskCoord.x.toFixed(3) + ", " 
                         + diskCoord.y.toFixed(3) + "]";
            distText = "";
        }
    }
    d3.select(".coordinates1").text(coordText1);
    d3.select(".coordinates2").text(coordText2);
    d3.select(".distance").text(distText);
}

function mousedown() {
    d3.event.preventDefault(); // disable text dragging

    canvasCoord = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
    diskCoord = canvasToDisk(canvasCoord)
    if (!diskCoord)
        return

    isMouseDown = true;
    srcCoord = canvasCoord;
}

function mousemove() {
    canvasCoord = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
    if (!canvasToDisk(canvasCoord)) {
        isMouseDown = false
        tmpObject = null
        drawObjects();
    } else if (isMouseDown 
               && canvasCoord != null 
               && canvasCoord.x != srcCoord.x
               && canvasCoord.y != srcCoord.y) {
        tmpObject = getTmpObject(srcCoord, canvasCoord)
    }

    if (canvasCoord != null) {
        drawInfoBox(canvasCoord);

        if (tmpObject != null)
            drawObjects();
    }
}

function mouseup() {
    isMouseDown = false
    if (tmpObject == null)
        return;

    if (drawType == "line") {
        canvasCoord = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
        diskCoord = canvasToDisk(canvasCoord)
        dist = hDist(canvasToDisk(tmpObject.p), canvasToDisk(tmpObject.q));

        if (diskCoord && tmpObject != null && dist > 0.0001) {
            tmpObject.isTmp = false;
            allObjects.push(tmpObject);
            drawElementsBox();
        }
    } else if (drawType == "circle") {
        canvasCoord = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
        diskCoord = canvasToDisk(canvasCoord)

        if (diskCoord && tmpObject != null && tmpObject.radius > 0.0001) {
            tmpObject.isTmp = false;
            allObjects.push(tmpObject)
            drawElementsBox()
        }

    }

    tmpObject = null
    highlightObject(null);
}

var g = svg.append("g");
svg.on("mousedown", mousedown)
   .on("mouseup", mouseup)
   .on("mousemove", mousemove)
