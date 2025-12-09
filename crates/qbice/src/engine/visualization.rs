//! HTML+JS visualization of the dependency graph using Cytoscape.js.
//!
//! This module provides functionality to export the query dependency graph
//! as an interactive HTML page.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Write as FmtWrite,
    fs,
    io::Write,
    path::Path,
};

use crate::{
    config::Config,
    engine::{Engine, meta::State},
    query::{DynQuery, Query, QueryID},
};

/// Information about a node in the dependency graph.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// The query ID of this node.
    pub id: QueryID,
    /// Debug representation of the query key.
    pub label: String,
    /// Whether this node is an input query.
    pub is_input: bool,
    /// The type name of the query.
    pub type_name: String,
    /// Debug representation of the computed result value.
    pub result: Option<String>,
}

/// Information about an edge in the dependency graph.
#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    /// The source query ID (caller).
    pub source: QueryID,
    /// The target query ID (callee).
    pub target: QueryID,
    /// Whether this dependency edge is dirty (needs revalidation).
    pub is_dirty: Option<bool>,
}

/// A snapshot of the dependency graph for visualization.
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// All nodes in the graph.
    pub nodes: Vec<NodeInfo>,
    /// All edges in the graph (caller -> callee, i.e., forward dependencies).
    pub edges: Vec<EdgeInfo>,
}

impl<C: Config> Engine<C> {
    /// Creates a snapshot of the dependency graph starting from a specific
    /// query, traversing only forward dependencies (callees).
    ///
    /// This captures all queries that the given query depends on, directly or
    /// transitively.
    ///
    /// This method takes `&mut self` to ensure no concurrent requests or
    /// modifications are made while traversing the database.
    ///
    /// # Arguments
    ///
    /// * `query` - The root query to start the traversal from.
    #[must_use]
    pub fn snapshot_graph_from<Q: Query>(
        &mut self,
        query: &Q,
    ) -> GraphSnapshot {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut visited: HashSet<QueryID> = HashSet::new();
        let mut queue: VecDeque<QueryID> = VecDeque::new();

        // Compute the root query ID
        let root_id = DynQuery::<C>::query_identifier(
            query,
            self.database.initial_seed(),
        );

        queue.push_back(root_id);

        // BFS traversal of forward dependencies
        while let Some(current_id) = queue.pop_front() {
            if !visited.insert(current_id) {
                continue;
            }

            // Get the query meta
            let Some(meta) = self.database.get_query_meta(&current_id) else {
                continue;
            };

            // Create node info
            let label = format!("{:?}", meta.original_key());
            let type_name =
                format!("{:?}", meta.original_key().stable_type_id());

            // Get the computed result if available
            let result = if let State::Computed(computed) = meta.state() {
                Some(format!("{:?}", computed.result()))
            } else {
                None
            };

            nodes.push(NodeInfo {
                id: current_id,
                label,
                is_input: meta.is_input(),
                type_name,
                result,
            });

            // Get forward dependencies (callees) from computed state
            if let State::Computed(computed) = meta.state() {
                for (callee_id, is_dirty) in
                    computed.callee_info().callee_order()
                {
                    // Add edge: current -> callee (forward dependency)
                    edges.push(EdgeInfo {
                        source: current_id,
                        target: callee_id,
                        is_dirty,
                    });

                    // Queue callee for traversal if not visited
                    if !visited.contains(&callee_id) {
                        queue.push_back(callee_id);
                    }
                }
            }
        }

        GraphSnapshot { nodes, edges }
    }

    /// Generates an interactive HTML visualization of the dependency graph
    /// starting from a specific query.
    ///
    /// The generated HTML file uses Cytoscape.js to render an interactive
    /// graph showing only forward dependencies (what the query depends on).
    /// Features include:
    /// - Pan and zoom the graph
    /// - Click nodes to see details
    /// - Search for specific queries
    /// - Filter by query type
    ///
    /// This method takes `&mut self` to ensure no concurrent requests or
    /// modifications are made while visualizing the database.
    ///
    /// # Arguments
    ///
    /// * `query` - The root query to start visualization from.
    /// * `output_path` - The path where the HTML file will be written.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn visualize_html<Q: Query>(
        &mut self,
        query: &Q,
        output_path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        let snapshot = self.snapshot_graph_from(query);
        write_html_visualization(&snapshot, output_path)
    }
}

/// Writes the HTML visualization to a file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_html_visualization(
    snapshot: &GraphSnapshot,
    output_path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let html = generate_html(snapshot);

    let path = output_path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(path)?;
    file.write_all(html.as_bytes())?;

    Ok(())
}

/// Escapes a string for use in JavaScript.
fn escape_js_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('<', "\\x3c")
        .replace('>', "\\x3e")
}

/// Converts a `QueryID` to a stable string identifier for Cytoscape.
fn query_id_to_string(id: &QueryID) -> String {
    // Use both stable_type_id and hash_128 to ensure uniqueness across types
    format!("q_{:032x}_{:032x}", id.stable_type_id().as_u128(), id.hash_128())
}

/// Generates the nodes JSON array for Cytoscape.
fn generate_nodes_json(
    nodes: &[NodeInfo],
    type_counts: &mut HashMap<String, usize>,
) -> String {
    let mut result = String::from("[");

    for (i, node) in nodes.iter().enumerate() {
        if i > 0 {
            result.push(',');
        }

        *type_counts.entry(node.type_name.clone()).or_insert(0) += 1;

        let node_class = if node.is_input { "input" } else { "computed" };
        let escaped_label = escape_js_string(&node.label);
        let escaped_type = escape_js_string(&node.type_name);
        let id_str = query_id_to_string(&node.id);
        let truncated = escape_js_string(&truncate_label(&node.label, 30));
        let escaped_result = node
            .result
            .as_ref()
            .map(|r| escape_js_string(r))
            .unwrap_or_default();

        write!(
            &mut result,
            r"{{ data: {{ id: '{id_str}', label: '{escaped_label}', typeName: '{escaped_type}', isInput: {is_input}, truncatedLabel: '{truncated}', result: '{escaped_result}' }}, classes: '{node_class}' }}",
            is_input = node.is_input,
        )
        .unwrap();
    }
    result.push(']');
    result
}

/// Generates the edges JSON array for Cytoscape.
fn generate_edges_json(edges: &[EdgeInfo]) -> String {
    let mut result = String::from("[");

    for (i, edge) in edges.iter().enumerate() {
        if i > 0 {
            result.push(',');
        }

        let source_str = query_id_to_string(&edge.source);
        let target_str = query_id_to_string(&edge.target);

        // Determine edge class based on dirty status
        let edge_class = match edge.is_dirty {
            Some(true) => "dirty",
            Some(false) => "clean",
            None => "unknown",
        };

        write!(
            &mut result,
            r"{{ data: {{ source: '{source_str}', target: '{target_str}' }}, classes: '{edge_class}' }}"
        )
        .unwrap();
    }
    result.push(']');
    result
}

/// Generates the type filter options HTML.
fn generate_type_options(type_counts: &HashMap<String, usize>) -> String {
    let mut result = String::new();
    let mut sorted_types: Vec<_> = type_counts.iter().collect();
    sorted_types.sort_by_key(|(name, _)| name.as_str());

    for (type_name, count) in sorted_types {
        let escaped = escape_js_string(type_name);
        write!(
            &mut result,
            r#"<option value="{escaped}">{escaped} ({count})</option>"#
        )
        .unwrap();
    }
    result
}

/// Generates the HTML content for the visualization.
///
/// This function is used internally and is exposed for testing.
#[allow(clippy::too_many_lines)]
#[cfg(test)]
pub(super) fn generate_html_for_test(snapshot: &GraphSnapshot) -> String {
    generate_html(snapshot)
}

/// Generates the HTML content for the visualization.
#[allow(clippy::too_many_lines)]
fn generate_html(snapshot: &GraphSnapshot) -> String {
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let nodes_json = generate_nodes_json(&snapshot.nodes, &mut type_counts);
    let edges_json = generate_edges_json(&snapshot.edges);
    let type_options = generate_type_options(&type_counts);

    let total_nodes = snapshot.nodes.len();
    let total_edges = snapshot.edges.len();
    let input_count = snapshot.nodes.iter().filter(|n| n.is_input).count();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QBICE Dependency Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e; color: #eee; height: 100vh; overflow: hidden;
        }}
        .container {{ display: flex; height: 100vh; }}
        #cy {{ flex: 1; background: #16213e; }}
        .sidebar {{
            width: 350px; background: #0f3460; padding: 20px;
            overflow-y: auto; border-left: 2px solid #e94560;
        }}
        h1 {{ font-size: 1.5rem; margin-bottom: 20px; color: #e94560; }}
        h2 {{ font-size: 1.1rem; margin: 20px 0 10px; color: #e94560; }}
        .stats {{
            background: rgba(233, 69, 96, 0.1); padding: 15px;
            border-radius: 8px; margin-bottom: 20px;
        }}
        .stat-item {{ display: flex; justify-content: space-between; padding: 5px 0; }}
        .stat-value {{ color: #e94560; font-weight: bold; }}
        .controls {{ margin-bottom: 20px; }}
        label {{ display: block; margin-bottom: 5px; font-size: 0.9rem; color: #aaa; }}
        input[type="text"], select {{
            width: 100%; padding: 10px; border: none; border-radius: 4px;
            background: #1a1a2e; color: #eee; margin-bottom: 10px;
        }}
        input[type="text"]:focus, select:focus {{ outline: 2px solid #e94560; }}
        button {{
            padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer;
            margin-right: 5px; margin-bottom: 5px; transition: all 0.2s;
        }}
        .btn-primary {{ background: #e94560; color: white; }}
        .btn-primary:hover {{ background: #ff6b6b; }}
        .btn-secondary {{ background: #1a1a2e; color: #eee; }}
        .btn-secondary:hover {{ background: #2a2a4e; }}
        .node-details {{
            background: #1a1a2e; padding: 15px; border-radius: 8px;
            margin-top: 20px; display: none;
        }}
        .node-details.active {{ display: block; }}
        .detail-label {{ color: #aaa; font-size: 0.8rem; text-transform: uppercase; margin-top: 10px; }}
        .detail-value {{ color: #eee; word-break: break-all; padding: 5px 0; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }}
        .badge-input {{ background: #4ecdc4; color: #1a1a2e; }}
        .badge-computed {{ background: #6c5ce7; color: white; }}
        .legend {{ margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px; }}
        .legend-item {{ display: flex; align-items: center; margin: 8px 0; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }}
        .legend-input {{ background: #4ecdc4; }}
        .legend-computed {{ background: #6c5ce7; }}
        .legend-line {{ width: 30px; height: 3px; margin-right: 10px; }}
        .legend-clean {{ background: #4ecdc4; }}
        .legend-dirty {{ background: #e94560; background: repeating-linear-gradient(90deg, #e94560, #e94560 5px, transparent 5px, transparent 10px); height: 3px; }}
        .legend-unknown {{ background: repeating-linear-gradient(90deg, #888, #888 3px, transparent 3px, transparent 6px); height: 3px; }}
        .layout-buttons {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div id="cy"></div>
        <div class="sidebar">
            <h1>🔍 QBICE Graph</h1>
            <div class="stats">
                <div class="stat-item"><span>Total Queries</span><span class="stat-value">{total_nodes}</span></div>
                <div class="stat-item"><span>Dependencies</span><span class="stat-value">{total_edges}</span></div>
                <div class="stat-item"><span>Input Queries</span><span class="stat-value">{input_count}</span></div>
            </div>
            <div class="controls">
                <label for="search">Search Queries</label>
                <input type="text" id="search" placeholder="Type to search...">
                <label for="type-filter">Filter by Type</label>
                <select id="type-filter"><option value="">All Types</option>{type_options}</select>
            </div>
            <h2>Layout</h2>
            <div class="layout-buttons">
                <button class="btn-primary" onclick="applyLayout('dagre')">Hierarchical</button>
                <button class="btn-secondary" onclick="applyLayout('cose')">Force</button>
                <button class="btn-secondary" onclick="applyLayout('breadthfirst')">Breadthfirst</button>
                <button class="btn-secondary" onclick="applyLayout('circle')">Circle</button>
                <button class="btn-secondary" onclick="applyLayout('grid')">Grid</button>
            </div>
            <h2>Actions</h2>
            <button class="btn-primary" onclick="cy.fit()">Fit to View</button>
            <button class="btn-secondary" onclick="cy.reset()">Reset</button>
            <button class="btn-secondary" onclick="highlightInputs()">Show Inputs</button>
            <button class="btn-secondary" onclick="clearHighlights()">Clear Highlights</button>
            <div id="node-details" class="node-details">
                <h2>Selected Query</h2>
                <div class="detail-label">Type</div>
                <div class="detail-value" id="detail-type"></div>
                <div class="detail-label">Label</div>
                <div class="detail-value" id="detail-label"></div>
                <div class="detail-label">Result</div>
                <div class="detail-value" id="detail-result"></div>
                <div class="detail-label">Kind</div>
                <div class="detail-value" id="detail-kind"></div>
                <div class="detail-label">Dependencies (Callees)</div>
                <div class="detail-value" id="detail-deps"></div>
                <div class="detail-label">Dependents (Callers)</div>
                <div class="detail-value" id="detail-dependents"></div>
                <button class="btn-secondary" style="margin-top: 10px;" onclick="highlightDependencies()">Highlight Dependencies</button>
                <button class="btn-secondary" onclick="highlightDependents()">Highlight Dependents</button>
            </div>
            <div class="legend">
                <h2>Legend</h2>
                <div class="legend-item"><div class="legend-color legend-input"></div><span>Input Query</span></div>
                <div class="legend-item"><div class="legend-color legend-computed"></div><span>Computed Query</span></div>
                <h3 style="margin-top: 15px; font-size: 0.9rem; color: #e94560;">Edges</h3>
                <div class="legend-item"><div class="legend-line legend-clean"></div><span>Clean (validated)</span></div>
                <div class="legend-item"><div class="legend-line legend-dirty"></div><span>Dirty (needs revalidation)</span></div>
                <div class="legend-item"><div class="legend-line legend-unknown"></div><span>Unknown</span></div>
            </div>
        </div>
    </div>
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        let selectedNode = null;
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: {{ nodes: nodes, edges: edges }},
            style: [
                {{ selector: 'node', style: {{
                    'label': 'data(truncatedLabel)', 'text-valign': 'center', 'text-halign': 'center',
                    'background-color': '#6c5ce7', 'color': '#fff', 'font-size': '10px',
                    'text-wrap': 'ellipsis', 'text-max-width': '100px', 'width': 'label', 'height': 'label',
                    'padding': '10px', 'shape': 'roundrectangle', 'border-width': 2, 'border-color': '#8b7ee7'
                }} }},
                {{ selector: 'node.input', style: {{ 'background-color': '#4ecdc4', 'border-color': '#6ee7de', 'shape': 'ellipse' }} }},
                {{ selector: 'edge', style: {{
                    'width': 2, 'line-color': '#4ecdc4', 'target-arrow-color': '#4ecdc4',
                    'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'opacity': 0.7
                }} }},
                {{ selector: 'edge.clean', style: {{ 'line-color': '#4ecdc4', 'target-arrow-color': '#4ecdc4' }} }},
                {{ selector: 'edge.dirty', style: {{ 'line-color': '#e94560', 'target-arrow-color': '#e94560', 'line-style': 'dashed' }} }},
                {{ selector: 'edge.unknown', style: {{ 'line-color': '#888', 'target-arrow-color': '#888', 'line-style': 'dotted' }} }},
                {{ selector: 'node:selected', style: {{ 'border-width': 4, 'border-color': '#ff6b6b' }} }},
                {{ selector: 'node.highlighted', style: {{ 'background-color': '#ffd93d', 'border-color': '#ffec8b' }} }},
                {{ selector: 'node.dependency', style: {{ 'background-color': '#ff6b6b', 'border-color': '#ff8c8c' }} }},
                {{ selector: 'node.dependent', style: {{ 'background-color': '#4ecdc4', 'border-color': '#6ee7de' }} }},
                {{ selector: 'edge.highlighted', style: {{ 'line-color': '#ffd93d', 'target-arrow-color': '#ffd93d', 'width': 4, 'opacity': 1 }} }},
                {{ selector: 'node.faded', style: {{ 'opacity': 0.2 }} }},
                {{ selector: 'edge.faded', style: {{ 'opacity': 0.1 }} }}
            ],
            layout: {{ name: 'dagre', rankDir: 'TB', nodeSep: 50, rankSep: 80, animate: true, animationDuration: 500 }},
            wheelSensitivity: 0.3
        }});
        cy.on('tap', 'node', function(evt) {{ selectedNode = evt.target; showNodeDetails(selectedNode); }});
        cy.on('tap', function(evt) {{ if (evt.target === cy) {{ hideNodeDetails(); selectedNode = null; }} }});
        function showNodeDetails(node) {{
            const data = node.data();
            document.getElementById('detail-type').textContent = data.typeName;
            document.getElementById('detail-label').textContent = data.label;
            document.getElementById('detail-result').textContent = data.result || '(not computed)';
            document.getElementById('detail-kind').innerHTML = data.isInput 
                ? '<span class="badge badge-input">Input</span>' 
                : '<span class="badge badge-computed">Computed</span>';
            document.getElementById('detail-deps').textContent = node.outgoers('node').length;
            document.getElementById('detail-dependents').textContent = node.incomers('node').length;
            document.getElementById('node-details').classList.add('active');
        }}
        function hideNodeDetails() {{ document.getElementById('node-details').classList.remove('active'); }}
        function applyLayout(name) {{
            let options = {{ name: name, animate: true, animationDuration: 500 }};
            if (name === 'dagre') {{ options.rankDir = 'TB'; options.nodeSep = 50; options.rankSep = 80; }}
            else if (name === 'cose') {{ options.nodeRepulsion = 400000; options.idealEdgeLength = 100; }}
            else if (name === 'breadthfirst') {{ options.directed = true; options.spacingFactor = 1.5; }}
            cy.layout(options).run();
        }}
        function highlightDependencies() {{
            if (!selectedNode) return;
            clearHighlights();
            const deps = selectedNode.successors();
            cy.elements().addClass('faded');
            selectedNode.removeClass('faded');
            deps.removeClass('faded');
            deps.nodes().addClass('dependency');
            deps.edges().addClass('highlighted');
        }}
        function highlightDependents() {{
            if (!selectedNode) return;
            clearHighlights();
            const dependents = selectedNode.predecessors();
            cy.elements().addClass('faded');
            selectedNode.removeClass('faded');
            dependents.removeClass('faded');
            dependents.nodes().addClass('dependent');
            dependents.edges().addClass('highlighted');
        }}
        function highlightInputs() {{
            clearHighlights();
            cy.elements().addClass('faded');
            cy.nodes('.input').removeClass('faded').addClass('highlighted');
        }}
        function clearHighlights() {{ cy.elements().removeClass('faded highlighted dependency dependent'); }}
        document.getElementById('search').addEventListener('input', function(e) {{
            const query = e.target.value.toLowerCase();
            if (!query) {{ clearHighlights(); return; }}
            clearHighlights();
            cy.elements().addClass('faded');
            cy.nodes().forEach(function(node) {{
                const label = node.data('label').toLowerCase();
                const typeName = node.data('typeName').toLowerCase();
                if (label.includes(query) || typeName.includes(query)) {{
                    node.removeClass('faded').addClass('highlighted');
                }}
            }});
        }});
        document.getElementById('type-filter').addEventListener('change', function(e) {{
            const selectedType = e.target.value;
            if (!selectedType) {{ clearHighlights(); return; }}
            clearHighlights();
            cy.elements().addClass('faded');
            cy.nodes().forEach(function(node) {{
                if (node.data('typeName') === selectedType) {{
                    node.removeClass('faded').addClass('highlighted');
                }}
            }});
        }});
    </script>
</body>
</html>
"#
    )
}

/// Truncates a label to a maximum length, adding ellipsis if needed.
fn truncate_label(label: &str, max_len: usize) -> String {
    if label.len() <= max_len {
        label.to_string()
    } else {
        format!("{}...", &label[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_js_string() {
        assert_eq!(escape_js_string("hello"), "hello");
        assert_eq!(escape_js_string("it's"), "it\\'s");
        assert_eq!(escape_js_string("line\nbreak"), "line\\nbreak");
        assert_eq!(escape_js_string("<script>"), "\\x3cscript\\x3e");
    }

    #[test]
    fn test_truncate_label() {
        assert_eq!(truncate_label("short", 10), "short");
        assert_eq!(
            truncate_label("this is a very long label", 10),
            "this is..."
        );
    }

    #[test]
    fn test_empty_graph_snapshot() {
        let snapshot = GraphSnapshot { nodes: vec![], edges: vec![] };
        let html = generate_html(&snapshot);
        assert!(html.contains("QBICE Dependency Graph"));
        assert!(html.contains("Total Queries"));
        assert!(html.contains(">0<")); // total nodes is 0
    }

    #[test]
    fn test_generate_type_options() {
        let mut type_counts = HashMap::new();
        type_counts.insert("TypeA".to_string(), 5);
        type_counts.insert("TypeB".to_string(), 3);

        let options = generate_type_options(&type_counts);

        assert!(options.contains("TypeA"));
        assert!(options.contains("TypeB"));
        assert!(options.contains("(5)"));
        assert!(options.contains("(3)"));
    }
}
