function InitializeTables(){
    $('.dataTables').dataTable({
        responsive: true,
    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bPaginate": false
    });
    $('.dataTables-nofilter').dataTable({
        responsive: true,
        "bPaginate": false,
        "bFilter": false
    });
}
