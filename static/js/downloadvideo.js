// jQuery(function($) {
//   var $form = $('#formDownload');
  
//   $("#buttonDownload").click(function(e) {
//     e.preventDefault();
    
//     if (!$form[0].checkValidity()) {
//       $form[0].reportValidity();
//     } else {
//       $("#form").submit();
//       var formData = new FormData($('#formSearch')[0]);
//       var inputValue = formData.get("searchName");
//       console.log(inputValue);
//       location.replace('./dataset?index=0&nameSearch=' + inputValue);
//     }
//   });
// });


$(function() {

  $('#buttonDownload').bind('click', function(event) {
    var form_data = new FormData($('#formDownload')[0]);
    // console.log('form_data', String(form_data.has('inputName')));
    // formData.has('inputName');
    if (form_data.get('inputName') == '' && form_data.get('inputURI') == '') {
      $('#respMessage').text('Please enter a name or a URI');
      show_and_hide_alert();
    } else {
    $.ajax({
      url: './downloadVideo',
      // data: $('#formDownload').serialize(),
      data: form_data,
      type: 'POST',
      contentType: false,
      cache: false,
      processData: false,
      async: true,
      success: function(response) {
        video_path = response['video_path']
        name_video = response['name_video']
        $('#respMessage').prepend('<span>Downloaded: <a href="static/' + '" download>'+ video_path + '</a></span><br>');
        show_and_hide_alert();
        console.log(video_path);
        addCamera(video_path);
      },
      error: function(err) {
        $('#respMessage').prepend('<span>Internal error!</span><br>');
        show_and_hide_alert();
      }
    });
  }
  });

  $('#inputURI').bind('keypress', function(event) {
    if(event.which == 13) {
      event.preventDefault();
      $('#buttonDownload').click();
    }
  });
  
});

function show_and_hide_alert() {
  $('#respMessage').fadeIn(300);
  setTimeout(function() { 
      $('#respMessage').fadeOut(600); 
  }, 5000);
}

$(function() {
  $('#btn-stream').click(function () {
    console.log('home page')
    window.open("./");
  })
});

$(function() {
  $('#btn-home').click(function () {
    console.log('home page')
    window.open("./home");
  })
});

$(function() {
  $('#btn-dataset').click(function () {
    console.log('dataset page')
    window.open("./dataset?index=0");
  })
});

$(document).ready(function(){
  $('.play').click(function () {
      if($(this).parent().prev().get(0).paused){
          $(this).parent().prev().get(0).play();
          // $(this).parent().prev().removeClass('blurEffect');
          // $('.content').hide();
      }
  });

  $('.video').on('ended',function(){
      // $(this).addClass('blurEffect');
    $('.content').show();
  });
})

