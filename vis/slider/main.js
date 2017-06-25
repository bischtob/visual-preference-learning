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
    $('#slider').slick('unslick');
  
    /* load html (asynchronous, wait for successful callback) */

    /* We hide visiblity to prevent the user from seeing 
     * the bare images flash before slick can run on them.
     * we can't use .hide() because that sets display:none,
     * which breaks slick.
     *
     * We use the callback on 'init' (see below) to 
     * make the div visible again.
     */
    $('#slider').css('visibility', 'hidden');

    $('#slider').load(fn, function() {

      // update slider
      $('#slider').slick({
        infinite: true,
        slidesToShow: 1,
        slidesToScroll: 1,
      });

      // update text
      $('#n1').html(1);
      $('#n2').html(k);
    });

  });

  // used for update after clicking a kselect button
  $('#slider').on('init', function() {
    $('#slider').css('visibility', 'visible');
  });
  
  $('#slider').on('afterChange', function (event, slide) {
    // we use 1-indexing for display
    var currentSlide = slide.currentSlide+1;
    console.log(currentSlide);

    // change the text to show currentSlide
    $('#n1').html(currentSlide);
  });
});
