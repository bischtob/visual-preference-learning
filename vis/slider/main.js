$(document).on('ready', function() {
  $('.regular').slick({
    infinite: true,
    slidesToShow: 1,
    slidesToScroll: 1,
  });

  $('.kselect').click(function(e) {
    var k = e.target.id;
    var fn = "centers/k"+k+"/list.html"; /* gives the file name*/
  
    /* destroy slick */
   $('.regular').slick('unslick');
  
    /* load html (asynchronous, wait for successful callback) */
    $('.regular').load(fn, function() {

      // update slider
      $('.regular').slick({
       infinite: true,
       slidesToShow: 1,
       slidesToScroll: 1,
     });

     // update text
     $('#n1').html(1);
     $('#n2').html(k);
    });
  
  });
  
  $('.regular').on('afterChange', function (event, slide) {
    // we use 1-indexing for display
    var currentSlide = slide.currentSlide+1;
    console.log(currentSlide);

    // change the text to show currentSlide
    $('#n1').html(currentSlide);
  });
});
