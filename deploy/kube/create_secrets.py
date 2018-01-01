import base64
import config

if __name__ == '__main__':
    with open('secrets_template.yml') as f:
        template = f.read()
    with open('secrets.yml','w') as out:
        out.write(template.format(
            dbusername=base64.encodestring(config.dbusername),
            dbpassword=base64.encodestring(config.dbpassword),
            rabbithost=base64.encodestring(config.rabbithost),
            rabbitpassword=base64.encodestring(config.rabbitpassword),
            rabbitusername=base64.encodestring(config.rabbitusername),
            awskey=base64.encodestring(config.awskey),
            awssecret=base64.encodestring(config.awssecret),
            secretkey=base64.encodestring(config.secretkey),
            mediabucket=base64.encodestring(config.mediabucket),
            mediaurl=base64.encodestring('http://{}.storage.googleapis.com/'.format(config.mediabucket)),
            superuser=base64.encodestring(config.superuser),
            superpass=base64.encodestring(config.superpass),
            superemail=base64.encodestring(config.superemail),
            cloudfsprefix=base64.encodestring(config.cloudfsprefix),
        ).replace('\n\n','\n'))