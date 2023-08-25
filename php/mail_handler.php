<?php
/* Configuration */
$mailto = 'nabid.aust37@gmail.com';
$subject = "[Contact Form]";

$name = strip_tags($_POST['first_name']);
$sub = strip_tags($_POST['sub']);
$email = strip_tags($_POST['email']);
$comments = strip_tags($_POST['message']);

// HTML for email to send submission details
$body = "
<br>
<p><b>Message</b>: $comments</p>
<p><b>Name</b>: $name <br>
<p><b>Subject</b>: $sub <br>
<b>Email</b>: $email<br>
";

$headers = "From: $name <$email> \r\n";
$headers .= "Reply-To: $email \r\n";
$headers .= "MIME-Version: 1.0\r\n";
$headers .= "Content-Type: text/html; charset=ISO-8859-1\r\n";
$headers2 = "From:" . $mailto;
$message = "<html><body>$body</body></html>";

if (empty($name) || empty($sub) || empty($email) || empty($comments)) {
    header("Location: https://nabidalam.github.io/"); // go to home page
    die();
} else {
    if (mail($mailto, $subject, $message, $headers)) {
        header("Location: https://nabidalam.github.io/?success=true#contact-form");
    } else {
        echo "Failed"; // failure
    }
}
?>
